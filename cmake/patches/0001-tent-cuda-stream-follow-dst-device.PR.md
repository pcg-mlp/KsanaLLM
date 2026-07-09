# Upstream PR Draft: Mooncake tent — `CudaPlatform::copy` cross-device safety

**Status**: Draft — to be submitted to https://github.com/kvcache-ai/Mooncake

**Branch base**: `main` (commit `368b41f1` `[Docs] Tag 3fs Feature as Experimental (#2062)`)

**Files changed**:
- `mooncake-transfer-engine/tent/src/platform/cuda/cuda_allocator.cpp`
- `mooncake-transfer-engine/tent/src/runtime/control_plane.cpp`

**Patch source**: `cmake/patches/0001-tent-cuda-stream-follow-dst-device.patch` in the
ksana_llm vendored copy.

---

## Title

`tent: fix Platform::getLoader().copy silent failure on cross-device GPU pointers`

## Body

### Problem

In a multi-GPU-per-process deployment of `mooncake-transfer-engine/tent` (TCP transport
or HOST-RDMA staging), the control-plane RPC server worker threads default to `cuda:0`
because no caller invokes `cudaSetDevice` on those threads. When the locally
registered KV/data buffer lives on `cuda:R` for `R != 0`, the receive-side
`onSendData` handler ends up calling

```cpp
Platform::getLoader().copy((void*)peer_mem_addr, &desc[1], length);
```

which lands in `CudaPlatform::copy`:

```cpp
CUDAStreamHandle stream;
CHECK_STATUS(getStreamFromPool(stream));   // deviceId = kCurrentDevice
cudaMemcpyAsync(dst, src, length, cudaMemcpyDefault, stream.get());
```

`getStreamFromPool` falls back to `cudaGetDevice()` of the calling thread (= `cuda:0`),
so the stream is on `cuda:0` while `dst` is on `cuda:R`. Without explicit peer access
the async memcpy fails with a silent CUDA error. Worse, `onSendData`
(`tent/src/runtime/control_plane.cpp`) **doesn't check the return value** of
`Platform::getLoader().copy()` — the RPC handler returns OK, the sender's
`getBatchTransferStatus` reports `COMPLETED`, but the destination GPU memory was
never written.

### Symptoms

In Tencent's pd_v2 disaggregated-prefill smoke (`scheduler_type=1`, Mooncake tent
TCP transport, DSV2-Lite-Chat MLA, TP=4 DP=1 + Decode rank R != 0):

- Decode rank 0 receives correct KV bytes ✓
- Decode rank 1/2/3 GPU memory stays all-zero ✗
- Mooncake reports all 4 transfers `COMPLETED`
- Decode forward sees zero KV at attention, all-reduce mixes garbage,
  output stutters: `"many many many many..."`

`MC_DISABLE_GPU_DIRECT_RDMA=1` (forces TCP) makes it deterministic. GDR-supported
hosts mask the bug because `ibv_post_send` doesn't go through `Platform::copy`.

### Fix

Two narrow changes to the existing files; no API change, no behavior regression on
single-GPU or already-correct (current device matches dst) call sites.

#### 1. `CudaPlatform::copy` — pin the copy to the dst device

```cpp
Status CudaPlatform::copy(void* dst, void* src, size_t length) {
    int target_dev = -1;
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, dst) == cudaSuccess &&
        attr.type == cudaMemoryTypeDevice) {
        target_dev = attr.device;
    } else if (cudaPointerGetAttributes(&attr, src) == cudaSuccess &&
               attr.type == cudaMemoryTypeDevice) {
        target_dev = attr.device;
    }

    int prev_dev = -1;
    if (target_dev >= 0) {
        CHECK_CUDA(cudaGetDevice(&prev_dev));
        if (prev_dev != target_dev) {
            CHECK_CUDA(cudaSetDevice(target_dev));
        }
    }

    cudaError_t cerr = cudaMemcpy(dst, src, length, cudaMemcpyDefault);

    if (target_dev >= 0 && prev_dev != target_dev && prev_dev >= 0) {
        cudaSetDevice(prev_dev);
    }

    if (cerr != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpy: " << cudaGetErrorString(cerr);
        return Status::InternalError("cudaMemcpy failed" LOC_MARK);
    }
    return Status::OK();
}
```

Switched from `cudaMemcpyAsync` + stream-pool to synchronous `cudaMemcpy` because
the call-sites (`SendData`, `RecvData`, staging local-segment transfers) are all
already synchronous from the caller's perspective (they call `cudaStreamSynchronize`
right after). The mooncake-pg deadlock concern in the original comment doesn't
apply on this code path. Sync `cudaMemcpy` + temporary `cudaSetDevice(target_dev)`
removes the cross-device stream constraint entirely.

The `cudaSetDevice` is restored before returning, leaving the calling thread's
state observably unchanged — same pattern as `cuda_alloc` in this file.

#### 2. `onSendData` / `onRecvData` — propagate copy failures

```cpp
if (local_desc->findBuffer(peer_mem_addr, length)) {
    auto status = Platform::getLoader().copy(
        (void*)peer_mem_addr, &desc[1], length);
    if (!status.ok()) {
        response = "SendData failed: copy: " + status.ToString();
    }
} else {
    response = "SendData failed: target address not in registered buffer";
}
```

The copy return value was previously discarded. Now a non-OK status writes an
RPC error response so the sender's `getBatchTransferStatus` can correctly report
`FAILED` instead of silently lying.

### Test

```
csrc/pd_v2/mooncake/k8a_cross_device_copy_test.cpp (gtest)
```

Two contract tests pin down the CUDA-runtime premises the fix relies on:

- `cudaPointerGetAttributes` resolves a `cudaMalloc`'d buffer to its owning
  device regardless of the calling thread's current device.
- Temporary `cudaSetDevice(target)` + sync `cudaMemcpy(cudaMemcpyDefault)`
  succeeds without peer access; restoring `cudaSetDevice(prev)` leaves the
  calling thread's CUDA state observably unchanged.

Skipped automatically when fewer than 2 visible GPUs.

### Integration verification

In ksana_llm pd_v2 disaggregated-prefill smoke:

- TCP mode `DSV2-Lite-Chat` `TP=4 DP=1` `chunk_size {1, 14, 27}` × short/long prompt:
  output byte-identical (greedy decode) across all combinations.
- GDR mode same config: byte-identical to TCP (sanity that fix doesn't regress
  GDR path which doesn't call `CudaPlatform::copy` anyway).
- Mode change: control-plane TCP path no longer silently drops bytes on
  `cuda:R != 0` Decode ranks.
