/* Copyright 2026 Tencent Inc.  All rights reserved.

   Kunlun XPU device traits / stream / event / memory primitives.
   Mirrors device/zixiao/zixiao_device.h.

   关键类型决策（基于 <xpu/runtime.h> 实测）：
     XPUStream / XPUEvent 各自 typedef 为 `void*`，**不是** cudaStream_t 的别名。
     选择 XPUStream 是为了与 BKCL 接口（bkcl_all_reduce 等）的形参类型一致。
==============================================================================*/
#pragma once

#include <cstddef>
#include <unordered_map>

#include "device/common_device.h"
#include "device/kunlun/xpu_utils.h"

namespace ksana_llm {

// Maximum single-shot xpu_memcpy(_async) size before we split into chunks.
// The XPU async DMA linklist descriptor holds ~87209 SG entries (~340 MB of
// scattered host pages at 4 KiB / page).  A single H2D/D2H larger than that
// with fragmented host memory overflows the descriptor and the driver returns
// NV_ERR_OUT_OF_RANGE / CUDA_ERROR_ASYNC_COPY_OUTPUT_OF_RANGE(911), surfaced
// upward as -999.  256 MiB caps entry usage at 65536 (~75% of capacity),
// leaving room for host page fragmentation.
inline constexpr size_t kMaxDmaChunkSize = 256ULL * 1024 * 1024;

class KunlunDeviceContextManager {
 public:
  KunlunDeviceContextManager();
  ~KunlunDeviceContextManager();
};

// --- Stream specialization ----------------------------------------------------
// 这是模板"全特化" (full specialization)：把 StreamTypeTraits<int> 的 int 槽位
// 钉死成 DEVICE_TYPE_KUNLUN，从而告诉框架"昆仑后端的 stream 内部类型是 XPUStream"。
// common_device.h 的 primary template 把 value_type 设为 void，特化后变成 XPUStream。
template <>
struct StreamTypeTraits<DEVICE_TYPE_KUNLUN> {
  typedef XPUStream value_type;
};

template <>
class StreamT<DEVICE_TYPE_KUNLUN> {
 public:
  StreamT();

  XPUStream& Get();

  // Destroy the stream.
  void Destroy();

 private:
  int device_id_;
  XPUStream xpu_stream_;
};

template <>
struct EventTypeTraits<DEVICE_TYPE_KUNLUN> {
  typedef XPUEvent value_type;
};

template <>
class EventT<DEVICE_TYPE_KUNLUN> {
 public:
  XPUEvent& Get();

 private:
  XPUEvent xpu_event_;
};

template <>
void EventCreateT<DEVICE_TYPE_KUNLUN>(EventT<DEVICE_TYPE_KUNLUN>* event);

template <>
void EventCreateWithFlagsT(EventT<DEVICE_TYPE_KUNLUN>* event, unsigned int flags);

template <>
void EventDestroyT<DEVICE_TYPE_KUNLUN>(EventT<DEVICE_TYPE_KUNLUN> event);

template <>
void StreamSynchronizeT<DEVICE_TYPE_KUNLUN>(StreamT<DEVICE_TYPE_KUNLUN> stream);

template <>
void DeviceSynchronizeT<DEVICE_TYPE_KUNLUN>();

template <>
void SetDeviceT<DEVICE_TYPE_KUNLUN>(int device_id);

template <>
void GetDeviceT<DEVICE_TYPE_KUNLUN>(int* device_id);

template <>
void GetDeviceCountT<DEVICE_TYPE_KUNLUN>(int* count);

template <>
void EventRecordT<DEVICE_TYPE_KUNLUN>(EventT<DEVICE_TYPE_KUNLUN> event, StreamT<DEVICE_TYPE_KUNLUN> stream);

template <>
void StreamWaitEventT<DEVICE_TYPE_KUNLUN>(StreamT<DEVICE_TYPE_KUNLUN> stream, EventT<DEVICE_TYPE_KUNLUN> event);

template <>
void EventSynchronizeT<DEVICE_TYPE_KUNLUN>(EventT<DEVICE_TYPE_KUNLUN> event);

template <>
void EventElapsedTimeT<DEVICE_TYPE_KUNLUN>(float* ms, EventT<DEVICE_TYPE_KUNLUN> start, EventT<DEVICE_TYPE_KUNLUN> end);

template <>
void MemGetInfoT<DEVICE_TYPE_KUNLUN>(size_t* free, size_t* total);

template <>
void MallocT<DEVICE_TYPE_KUNLUN>(void** dev_ptr, size_t size);

template <>
void FreeT<DEVICE_TYPE_KUNLUN>(void* dev_ptr);

template <>
void HostAllocT<DEVICE_TYPE_KUNLUN>(void** host_ptr, size_t size);

template <>
void FreeHostT<DEVICE_TYPE_KUNLUN>(void* host_ptr);

template <>
void HostAllocMappedT<DEVICE_TYPE_KUNLUN>(void** host_ptr, void** device_ptr, size_t size);

template <>
void FreeHostMappedT<DEVICE_TYPE_KUNLUN>(void* host_ptr, void* device_ptr);

template <>
void MallocAsyncT<DEVICE_TYPE_KUNLUN>(void** dev_ptr, size_t size, StreamT<DEVICE_TYPE_KUNLUN> stream);

template <>
void FreeAsyncT<DEVICE_TYPE_KUNLUN>(void* dev_ptr, StreamT<DEVICE_TYPE_KUNLUN> stream);

template <>
void MemsetAsyncT<DEVICE_TYPE_KUNLUN>(void* dev_ptr, int value, size_t count, StreamT<DEVICE_TYPE_KUNLUN> stream);

template <>
void MemsetT<DEVICE_TYPE_KUNLUN>(void* dev_ptr, int value, size_t count);

template <>
void MemcpyAsyncT<DEVICE_TYPE_KUNLUN>(void* dst, const void* src, size_t count, enum MemcpyKind kind,
                                      StreamT<DEVICE_TYPE_KUNLUN> stream);

template <>
void Memcpy2DT<DEVICE_TYPE_KUNLUN>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                   size_t height, enum MemcpyKind kind);

template <>
void Memcpy2DAsyncT<DEVICE_TYPE_KUNLUN>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                        size_t height, enum MemcpyKind kind, StreamT<DEVICE_TYPE_KUNLUN> stream);

template <>
void MemcpyT<DEVICE_TYPE_KUNLUN>(void* dst, const void* src, size_t count, enum MemcpyKind kind);

template <>
class GetDataTypeT<DEVICE_TYPE_KUNLUN> {
 public:
  template <class U>
  static DataType impl();
};

template <>
void* GetRuntimeContextT<DEVICE_TYPE_KUNLUN>();

}  // namespace ksana_llm
