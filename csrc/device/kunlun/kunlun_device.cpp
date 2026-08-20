/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "device/kunlun/kunlun_device.h"

#include <xpu/xpuml.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "device/device_utils.h"
#include "utils/logger.h"

namespace ksana_llm {

namespace {

// Slice `count` bytes into kMaxDmaChunkSize-sized pieces and invoke
// `copy_chunk(dst_i, src_i, chunk_i)` for each — shared by the sync/async
// Memcpy paths so we don't duplicate the chunking loop.
template <typename CopyFn>
inline void SplitAndCopyDma(void* dst, const void* src, size_t count, CopyFn&& copy_chunk) {
  for (size_t offset = 0; offset < count; offset += kMaxDmaChunkSize) {
    size_t chunk = std::min(kMaxDmaChunkSize, count - offset);
    copy_chunk(static_cast<char*>(dst) + offset, static_cast<const char*>(src) + offset, chunk);
  }
}

}  // namespace

static KunlunDeviceContextManager g_context_manager;

KunlunDeviceContextManager::KunlunDeviceContextManager() {}
KunlunDeviceContextManager::~KunlunDeviceContextManager() {}

// ---------- Stream ----------
StreamT<DEVICE_TYPE_KUNLUN>::StreamT() {
  XPU_CHECK(xpu_current_device(&device_id_));
  XPU_CHECK(xpu_stream_create(&xpu_stream_));
}

void StreamT<DEVICE_TYPE_KUNLUN>::Destroy() { XPU_CHECK(xpu_stream_destroy(xpu_stream_)); }

XPUStream& StreamT<DEVICE_TYPE_KUNLUN>::Get() { return xpu_stream_; }

// ---------- Event ----------
XPUEvent& EventT<DEVICE_TYPE_KUNLUN>::Get() { return xpu_event_; }

template <>
void EventCreateT<DEVICE_TYPE_KUNLUN>(EventT<DEVICE_TYPE_KUNLUN>* event) {
  XPU_CHECK(xpu_event_create(&(event->Get())));
}

template <>
void EventCreateWithFlagsT(EventT<DEVICE_TYPE_KUNLUN>* event, unsigned int flags) {
  // XPU runtime API 不接受 flags 参数 (与 cudaEventCreateWithFlags 不同)，
  // 这里直接忽略 flags，与 ZiXiao 同样处理。
  (void)flags;
  XPU_CHECK(xpu_event_create(&(event->Get())));
}

template <>
void EventDestroyT<DEVICE_TYPE_KUNLUN>(EventT<DEVICE_TYPE_KUNLUN> event) {
  XPU_CHECK(xpu_event_destroy(event.Get()));
}

template <>
void StreamSynchronizeT<DEVICE_TYPE_KUNLUN>(StreamT<DEVICE_TYPE_KUNLUN> stream) {
  // xpu_wait(stream) 阻塞等待该 stream 的所有任务完成。
  XPU_CHECK(xpu_wait(stream.Get()));
}

template <>
void DeviceSynchronizeT<DEVICE_TYPE_KUNLUN>() {
  // xpu_wait(NULL) 同步当前 device 的所有任务（runtime.h L397 的语义约定）。
  XPU_CHECK(xpu_wait(nullptr));
}

template <>
void SetDeviceT<DEVICE_TYPE_KUNLUN>(int device_id) {
  XPU_CHECK(xpu_set_device(device_id));
}

template <>
void GetDeviceT<DEVICE_TYPE_KUNLUN>(int* device_id) {
  XPU_CHECK(xpu_current_device(device_id));
}

template <>
void GetDeviceCountT<DEVICE_TYPE_KUNLUN>(int* count) {
  XPU_CHECK(xpu_device_count(count));
}

template <>
void EventRecordT<DEVICE_TYPE_KUNLUN>(EventT<DEVICE_TYPE_KUNLUN> event, StreamT<DEVICE_TYPE_KUNLUN> stream) {
  XPU_CHECK(xpu_event_record(event.Get(), stream.Get()));
}

template <>
void StreamWaitEventT<DEVICE_TYPE_KUNLUN>(StreamT<DEVICE_TYPE_KUNLUN> stream, EventT<DEVICE_TYPE_KUNLUN> event) {
  XPU_CHECK(xpu_stream_wait_event(stream.Get(), event.Get()));
}

template <>
void EventSynchronizeT<DEVICE_TYPE_KUNLUN>(EventT<DEVICE_TYPE_KUNLUN> event) {
  XPU_CHECK(xpu_event_wait(event.Get()));
}

template <>
void EventElapsedTimeT<DEVICE_TYPE_KUNLUN>(float* ms, EventT<DEVICE_TYPE_KUNLUN> start,
                                           EventT<DEVICE_TYPE_KUNLUN> end) {
  // XPU does not expose an event elapsed time API; return 0 to avoid divide-by-zero.
  (void)start;
  (void)end;
  if (ms) *ms = 0.0f;
}

// ---------- Memory ----------
// Ksana 通用 MemcpyKind → XRE XPUMemcpyKind 翻译（对齐 Ascend GetAclMemcpyKind 模式）。
// 注意：XPUMemcpyKind 无 HOST_TO_HOST 枚举，调用方需要在上层先判并走 std::memcpy。
static XPUMemcpyKind GetXpuMemcpyKind(enum MemcpyKind kind) {
  switch (kind) {
    case MEMCPY_HOST_TO_DEVICE:
      return XPU_HOST_TO_DEVICE;
    case MEMCPY_DEVICE_TO_HOST:
      return XPU_DEVICE_TO_HOST;
    case MEMCPY_DEVICE_TO_DEVICE:
      return XPU_DEVICE_TO_DEVICE;
    default:
      // MEMCPY_HOST_TO_HOST 不应进来（上层已分流）；兜底返 D2D 避免未初始化返回值。
      return XPU_DEVICE_TO_DEVICE;
  }
}

// 以 upper_bound 为上界，二分探测当前设备能够一次性成功 xpu_malloc 的最大字节数。
//
// 背景：xpumlDeviceGetMemoryInfo 返回的 freeGlobalMemory 是所有空闲内存的总和，
// 而非最大单次连续可分配量。昆仑芯未实现 VMM 虚拟内存映射，碎片化的空闲块无法
// 被拼接成一整块连续内存，因此实测 free=39616 MB 时一次性 malloc 34700 MB 即 OOM(-705)。
//
// 由于 KV cache 分配路径 (PreAllocateBlocks) 要求一次性连续分配整块内存，
// "最大单次连续可分配量" 才是真正约束上层 block_num 计算的值。这里以 API 返回的
// free 作为二分上界（免去从 1MB 指数增长找上界的阶段），向下收敛。
//
// kProbeThreshold 为二分停止时残余区间宽度：既是精度损失上界，也天然充当安全裕量
// （探测与真实分配是两次独立事件，留出裕量吸收期间的碎片抖动）。256MB 相对典型
// 数十 GB 的空闲量损失 <1%，同时相比 1MB 阈值约减半失败的 xpu_malloc 次数
// （每次失败既慢又会在终端刷 "cuapiMemAlloc failed" 日志）。
static size_t ProbeMaxContiguousMalloc(size_t upper_bound) {
  const size_t kProbeThreshold = 256ULL << 20;  // 256 MB

  void* probe_ptr = nullptr;
  auto can_alloc = [&](size_t n) -> bool {
    if (n == 0) return true;
    if (xpu_malloc(&probe_ptr, n, XPU_MEM_MAIN) != 0) return false;
    xpu_free(probe_ptr);
    probe_ptr = nullptr;
    return true;
  };

  // 无碎片的理想情况：直接分配上界即可成功，省去二分。
  if (can_alloc(upper_bound)) return upper_bound;

  // 二分：lo 恒为可分配值（初始 0 一定成功），hi 恒为失败值（upper_bound 已失败）。
  size_t lo = 0, hi = upper_bound;
  while (hi - lo > kProbeThreshold) {
    size_t mid = lo + (hi - lo) / 2;
    if (can_alloc(mid)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <>
void MemGetInfoT<DEVICE_TYPE_KUNLUN>(size_t* free, size_t* total) {
  // 先通过 xpuml API 获取空闲显存总量作为上界，再二分探测真正的最大连续可分配量。
  // 详见 ProbeMaxContiguousMalloc 的说明及 xpuml freeGlobalMemory 碎片化问题。

  xpumlInit();

  // 逻辑 ID 仅用于下方日志；正确的 xpuml 句柄解析（逻辑→物理→xpuml 序号）
  // 统一收敛到 GetXpumlHandleForCurrentDevice()，与单测共用同一份映射逻辑。
  // 让 helper 顺带回传 logical_id，避免此处再单独调一次 xpu_current_device——
  // 多线程 xpu_set_device 切换时两次读取可能落到不同设备，导致日志 logical_id
  // 与实际被查询的设备不一致（TOCTOU 竞态）。
  int device_id = 0;
  xpumlDevice_t device = GetXpumlHandleForCurrentDevice(&device_id);

  xpumlMemory_t memory;
  xpumlReturn_t ret = xpumlDeviceGetMemoryInfo(device, &memory);
  if (ret != XPUML_SUCCESS) {
    throw std::runtime_error("xpumlDeviceGetMemoryInfo failed: " + std::to_string(ret));
  }

  if (total) *total = static_cast<size_t>(memory.totalGlobalMemory);

  // Fallback: API 返回的 free 是空闲总量，碎片化 + 无 VMM 时无法一次性连续分配。
  // 以其为上界二分探测真正的最大连续可分配量，作为 free 返回给上层。
  // TODO(zhongzhicao): 后续昆仑芯提供虚拟显存映射，支持物理地址非连续的分配后移除
  size_t api_free = static_cast<size_t>(memory.freeGlobalMemory);
  size_t max_contiguous = ProbeMaxContiguousMalloc(api_free);
  if (free) *free = max_contiguous;

  int physical_id = -1;
  xpumlDeviceGetId(device, &physical_id);  // 仅用于日志，失败保持 -1
  KLLM_LOG_INFO << "MemGetInfoT<Kunlun>: logical_id=" << device_id << " physical_id=" << physical_id
                << " total_MB=" << (memory.totalGlobalMemory / (1024 * 1024))
                << " api_free_MB=" << (api_free / (1024 * 1024))
                << " probed_free_MB=" << (max_contiguous / (1024 * 1024))
                << " used_MB=" << (memory.usedGlobalMemory / (1024 * 1024));
}

template <>
void MallocT<DEVICE_TYPE_KUNLUN>(void** dev_ptr, size_t size) {
  XPU_CHECK(xpu_malloc(dev_ptr, size, XPU_MEM_MAIN));
}

template <>
void FreeT<DEVICE_TYPE_KUNLUN>(void* dev_ptr) {
  XPU_CHECK(xpu_free(dev_ptr));
}

template <>
void HostAllocT<DEVICE_TYPE_KUNLUN>(void** host_ptr, size_t size) {
  // xpu_host_alloc flags 参数在当前 SDK 是 placeholder（defs.h L171），固定传 0。
  XPU_CHECK(xpu_host_alloc(host_ptr, size, 0));
}

template <>
void FreeHostT<DEVICE_TYPE_KUNLUN>(void* host_ptr) {
  XPU_CHECK(xpu_host_free(host_ptr));
}

template <>
void HostAllocMappedT<DEVICE_TYPE_KUNLUN>(void** host_ptr, void** device_ptr, size_t size) {
  throw std::runtime_error("HostAllocMappedT is not supported on Kunlun XPU.");
  (void)host_ptr;
  (void)device_ptr;
  (void)size;
}

template <>
void FreeHostMappedT<DEVICE_TYPE_KUNLUN>(void* host_ptr, void* device_ptr) {
  throw std::runtime_error("FreeHostMappedT is not supported on Kunlun XPU.");
  (void)host_ptr;
  (void)device_ptr;
}

template <>
void MallocAsyncT<DEVICE_TYPE_KUNLUN>(void** dev_ptr, size_t size, StreamT<DEVICE_TYPE_KUNLUN> stream) {
  // XRE has no async-malloc; fall back to synchronous xpu_malloc (same as Ascend).
  (void)stream;
  XPU_CHECK(xpu_malloc(dev_ptr, size, XPU_MEM_MAIN));
}

template <>
void FreeAsyncT<DEVICE_TYPE_KUNLUN>(void* dev_ptr, StreamT<DEVICE_TYPE_KUNLUN> stream) {
  (void)stream;
  XPU_CHECK(xpu_free(dev_ptr));
}

template <>
void MemsetAsyncT<DEVICE_TYPE_KUNLUN>(void* dev_ptr, int value, size_t count, StreamT<DEVICE_TYPE_KUNLUN> stream) {
  if (count == 0) return;
  throw std::runtime_error("MemsetAsyncT is not supported on Kunlun XPU.");
  (void)dev_ptr;
  (void)value;
  (void)stream;
}

template <>
void MemsetT<DEVICE_TYPE_KUNLUN>(void* dev_ptr, int value, size_t count) {
  if (count == 0) return;
  throw std::runtime_error("MemsetT is not supported on Kunlun XPU.");
  (void)dev_ptr;
  (void)value;
}

template <>
void MemcpyAsyncT<DEVICE_TYPE_KUNLUN>(void* dst, const void* src, size_t count, enum MemcpyKind kind,
                                      StreamT<DEVICE_TYPE_KUNLUN> stream) {
  if (count == 0) return;
  if (kind == MEMCPY_HOST_TO_HOST) {
    // XRE XPUMemcpyKind 无 H2H 枚举；host 侧直接拷。
    std::memcpy(dst, src, count);
    return;
  }
  // See kMaxDmaChunkSize in kunlun_device.h for the SG-entry rationale.
  if (count > kMaxDmaChunkSize) {
    KLLM_LOG_INFO << "MemcpyAsyncT: splitting " << count << " bytes into "
                  << ((count + kMaxDmaChunkSize - 1) / kMaxDmaChunkSize) << " chunks";
    SplitAndCopyDma(dst, src, count, [&](void* d, const void* s, size_t n) {
      XPU_CHECK(xpu_memcpy_async(d, s, n, GetXpuMemcpyKind(kind), stream.Get()));
    });
    return;
  }
  XPU_CHECK(xpu_memcpy_async(dst, src, count, GetXpuMemcpyKind(kind), stream.Get()));
}

// Forward-declare the explicit specialization: Memcpy2DT below uses it as its
// contiguous fast path, but its definition appears later in this file.
template <>
void MemcpyT<DEVICE_TYPE_KUNLUN>(void* dst, const void* src, size_t count, enum MemcpyKind kind);

namespace {
// A 2D region degenerates into one contiguous block when both row strides equal the
// copied row width (or when there is only a single row). In that case the whole
// transfer can be issued as a single 1D DMA instead of `height` separate ones.
//
// This matters a lot on XPU: every xpu_memcpy[_async] is a separate DMA submission
// with fixed per-call overhead, so a `height`-row loop costs O(height) launches even
// when the data is fully contiguous. xsglang issues these as `memcpy_1d`, which is
// why its per-copy cost is ~0.74us vs ~1.10us x N rows on our side.
inline bool Is2DContiguous(size_t dpitch, size_t spitch, size_t width, size_t height) {
  return height == 1 || (dpitch == width && spitch == width);
}
}  // namespace

template <>
void Memcpy2DT<DEVICE_TYPE_KUNLUN>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                   size_t height, enum MemcpyKind kind) {
  // 2D memcpy: copy `height` rows, each `width` bytes, with src/dst row strides.
  // XPU has no native 2D memcpy; emulate with per-row copies.
  if (width == 0 || height == 0) return;
  // Fast path: fully contiguous region -> a single 1D copy.
  // MemcpyT also handles the kMaxDmaChunkSize splitting for us.
  if (Is2DContiguous(dpitch, spitch, width, height)) {
    MemcpyT<DEVICE_TYPE_KUNLUN>(dst, src, width * height, kind);
    return;
  }
  if (kind == MEMCPY_HOST_TO_HOST) {
    for (size_t row = 0; row < height; ++row) {
      std::memcpy(static_cast<char*>(dst) + row * dpitch, static_cast<const char*>(src) + row * spitch, width);
    }
    return;
  }
  // Synchronous: use xpu_memcpy per row
  for (size_t row = 0; row < height; ++row) {
    XPU_CHECK(xpu_memcpy(static_cast<char*>(dst) + row * dpitch, static_cast<const char*>(src) + row * spitch, width,
                         GetXpuMemcpyKind(kind)));
  }
}

template <>
void Memcpy2DAsyncT<DEVICE_TYPE_KUNLUN>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                        size_t height, enum MemcpyKind kind, StreamT<DEVICE_TYPE_KUNLUN> stream) {
  // Async 2D memcpy: emulate with per-row async copies.
  if (width == 0 || height == 0) return;
  // Fast path: fully contiguous region -> a single 1D async copy.
  if (Is2DContiguous(dpitch, spitch, width, height)) {
    MemcpyAsyncT<DEVICE_TYPE_KUNLUN>(dst, src, width * height, kind, stream);
    return;
  }
  if (kind == MEMCPY_HOST_TO_HOST) {
    for (size_t row = 0; row < height; ++row) {
      std::memcpy(static_cast<char*>(dst) + row * dpitch, static_cast<const char*>(src) + row * spitch, width);
    }
    return;
  }
  for (size_t row = 0; row < height; ++row) {
    XPU_CHECK(xpu_memcpy_async(static_cast<char*>(dst) + row * dpitch, static_cast<const char*>(src) + row * spitch,
                               width, GetXpuMemcpyKind(kind), stream.Get()));
  }
}

template <>
void MemcpyT<DEVICE_TYPE_KUNLUN>(void* dst, const void* src, size_t count, enum MemcpyKind kind) {
  if (count == 0) return;
  if (kind == MEMCPY_HOST_TO_HOST) {
    std::memcpy(dst, src, count);
    return;
  }
  // Same SG-entry-limit rationale as MemcpyAsyncT; see kunlun_device.h.
  if (count > kMaxDmaChunkSize) {
    KLLM_LOG_INFO << "MemcpyT: splitting " << count << " bytes into "
                  << ((count + kMaxDmaChunkSize - 1) / kMaxDmaChunkSize) << " chunks";
    SplitAndCopyDma(dst, src, count,
                    [&](void* d, const void* s, size_t n) { XPU_CHECK(xpu_memcpy(d, s, n, GetXpuMemcpyKind(kind))); });
    return;
  }
  XPU_CHECK(xpu_memcpy(dst, src, count, GetXpuMemcpyKind(kind)));
}

// ---------- DataType mapping ----------
// 镜像 ZiXiao 的 GetDataTypeT 实现 + 显式实例化。
template <class U>
DataType GetDataTypeT<DEVICE_TYPE_KUNLUN>::impl() {
  if (std::is_same<U, float>::value || std::is_same<U, const float>::value) {
    return TYPE_FP32;
  } else if (std::is_same<U, float16>::value || std::is_same<U, const float16>::value) {
    return TYPE_FP16;
  } else if (std::is_same<U, int32_t>::value || std::is_same<U, const int32_t>::value) {
    return TYPE_INT32;
  } else if (std::is_same<U, int64_t>::value || std::is_same<U, const int64_t>::value) {
    return TYPE_INT64;
  } else if (std::is_same<U, int8_t>::value || std::is_same<U, const int8_t>::value) {
    return TYPE_INT8;
  } else if (std::is_same<U, uint8_t>::value || std::is_same<U, const uint8_t>::value) {
    return TYPE_UINT8;
  } else if (std::is_same<U, uint32_t>::value || std::is_same<U, const uint32_t>::value) {
    return TYPE_UINT32;
  } else if (std::is_same<U, uint64_t>::value || std::is_same<U, const uint64_t>::value) {
    return TYPE_UINT64;
  } else if (std::is_same<U, bool>::value || std::is_same<U, const bool>::value) {
    return TYPE_BOOL;
  } else if (std::is_same<U, char>::value || std::is_same<U, const char>::value) {
    return TYPE_BYTES;
  } else if (std::is_pointer<U>::value) {
    return TYPE_POINTER;
  } else {
    return TYPE_INVALID;
  }
}

// 显式实例化 (explicit instantiation): 让模板成员函数的代码真正生成到 .o 里，
// 否则 .h 看不到 impl<U>() 主体的别的 TU 链接时会找不到符号。
template DataType GetDataTypeT<DEVICE_TYPE_KUNLUN>::impl<float>();
template DataType GetDataTypeT<DEVICE_TYPE_KUNLUN>::impl<float16>();
template DataType GetDataTypeT<DEVICE_TYPE_KUNLUN>::impl<int32_t>();
template DataType GetDataTypeT<DEVICE_TYPE_KUNLUN>::impl<int8_t>();
template DataType GetDataTypeT<DEVICE_TYPE_KUNLUN>::impl<uint8_t>();
template DataType GetDataTypeT<DEVICE_TYPE_KUNLUN>::impl<uint32_t>();
template DataType GetDataTypeT<DEVICE_TYPE_KUNLUN>::impl<uint64_t>();
template DataType GetDataTypeT<DEVICE_TYPE_KUNLUN>::impl<bool>();
template DataType GetDataTypeT<DEVICE_TYPE_KUNLUN>::impl<char>();

template <>
void* GetRuntimeContextT<DEVICE_TYPE_KUNLUN>() {
  return nullptr;
}

}  // namespace ksana_llm
