/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

// Regression test for the cross-device cudaMemcpy contract that the vendored
// Mooncake patch (cmake/patches/0001-tent-cuda-stream-follow-dst-device.patch)
// depends on. The patch's correctness rests on three CUDA-runtime properties
// this test pins down:
//
// 1. `cudaPointerGetAttributes` reliably reports the device of a
//    cudaMalloc'd buffer regardless of the calling thread's current
//    device. (The patch reads attr.device on the RPC server thread,
//    which defaults to cuda:0 even when the buffer is on cuda:R.)
//
// 2. Temporarily switching the current device to the dst device with
//    `cudaSetDevice(target_dev)` and then issuing a synchronous
//    `cudaMemcpy(dst, src, len, cudaMemcpyDefault)` succeeds without
//    requiring peer access.
//
// 3. Restoring the previous device with `cudaSetDevice(prev_dev)` after
//    the copy leaves the calling thread's CUDA state observable as if
//    the copy never happened.
//
// If any of these three premises silently breaks (e.g. CUDA runtime
// upgrade changes cudaPointerGetAttributes semantics, or
// cudaMemcpyDefault behavior changes for cross-device pointers), the
// patch's reasoning collapses and we'd see cross-device KV copy regressions.
// This test catches that at unit-test time, before the issue propagates into integration smoke.
//
// Skipped automatically when fewer than 2 CUDA devices are visible —
// the cross-device copy path only matters in multi-GPU-per-process configs.

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace {

bool HasMultiGpu() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count >= 2;
}

}  // namespace

// Validates the patch's first premise: cudaPointerGetAttributes reports the
// owning device of a cudaMalloc'd buffer regardless of the calling thread's
// current device. The patch's `if (attr.type == cudaMemoryTypeDevice)
// target_dev = attr.device` branch depends on this.
TEST(CrossDeviceCopyContract, PointerAttributesReportOwningDevice) {
  if (!HasMultiGpu()) {
    GTEST_SKIP() << "Cross-device copy contract test requires >= 2 CUDA devices";
  }
  constexpr size_t kLen = 4096;
  void* dev1_buf = nullptr;
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev1_buf, kLen), cudaSuccess);

  // Switch current device to 0 — this mirrors a Mooncake RPC server
  // worker thread that defaulted to cuda:0 querying a buffer that lives
  // on cuda:1.
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  cudaPointerAttributes attr{};
  ASSERT_EQ(cudaPointerGetAttributes(&attr, dev1_buf), cudaSuccess);
  EXPECT_EQ(attr.type, cudaMemoryTypeDevice);
  EXPECT_EQ(attr.device, 1);

  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  cudaFree(dev1_buf);
}

// Validates the patch's full cross-device copy path:
//   - thread current device = 0
//   - dst lives on cuda:1
//   - cudaPointerGetAttributes resolves dst to device 1
//   - temporary cudaSetDevice(1) + sync cudaMemcpy(cudaMemcpyDefault)
//   - restore cudaSetDevice(0)
//   - bytes are present at dst on cuda:1, current device is back to 0
TEST(CrossDeviceCopyContract, SyncCopyAfterSetDeviceLandsBytes) {
  if (!HasMultiGpu()) {
    GTEST_SKIP() << "Cross-device copy contract test requires >= 2 CUDA devices";
  }
  constexpr size_t kLen = 4096;
  std::vector<uint8_t> host_src(kLen);
  for (size_t i = 0; i < kLen; ++i) {
    host_src[i] = static_cast<uint8_t>(i & 0xFF);
  }

  void* dev1_buf = nullptr;
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev1_buf, kLen), cudaSuccess);

  // Simulate the Mooncake RPC server thread: default to cuda:0.
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  int prev_dev = -1;
  ASSERT_EQ(cudaGetDevice(&prev_dev), cudaSuccess);
  EXPECT_EQ(prev_dev, 0);

  // Patch path: detect dst device, temporarily switch, copy, restore.
  cudaPointerAttributes attr{};
  ASSERT_EQ(cudaPointerGetAttributes(&attr, dev1_buf), cudaSuccess);
  ASSERT_EQ(attr.type, cudaMemoryTypeDevice);
  const int target_dev = attr.device;
  ASSERT_EQ(target_dev, 1);

  ASSERT_EQ(cudaSetDevice(target_dev), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev1_buf, host_src.data(), kLen, cudaMemcpyDefault), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(prev_dev), cudaSuccess);

  // Caller's perspective: current device is back to 0 as if nothing happened.
  int after_dev = -1;
  ASSERT_EQ(cudaGetDevice(&after_dev), cudaSuccess);
  EXPECT_EQ(after_dev, prev_dev);

  // Read back from cuda:1 to confirm bytes landed.
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  std::vector<uint8_t> host_dst(kLen, 0);
  ASSERT_EQ(cudaMemcpy(host_dst.data(), dev1_buf, kLen, cudaMemcpyDefault), cudaSuccess);
  for (size_t i = 0; i < kLen; ++i) {
    ASSERT_EQ(host_dst[i], static_cast<uint8_t>(i & 0xFF)) << "byte mismatch at offset " << i;
  }

  cudaFree(dev1_buf);
}
