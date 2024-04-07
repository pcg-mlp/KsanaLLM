/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/ascend/ascend_device.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

StreamT<DEVICE_TYPE_ASCEND>::StreamT(int device_id) : device_id_(device_id) {
  ACL_CHECK(aclrtSetDevice(device_id_));
  ACL_CHECK(aclrtCreateStream(&acl_stream_));
}

void StreamT<DEVICE_TYPE_ASCEND>::Destroy() {
  ACL_CHECK(aclrtSetDevice(device_id_));
  ACL_CHECK(aclrtDestroyStream(acl_stream_));
}

aclrtStream& StreamT<DEVICE_TYPE_ASCEND>::Get() { return acl_stream_; }

aclrtEvent& EventT<DEVICE_TYPE_ASCEND>::Get() { return acl_event_; }

template <>
void EventCreateT<DEVICE_TYPE_ASCEND>(EventT<DEVICE_TYPE_ASCEND>* event) {
  // Default enable multiple stream sync, and enable timeline.
  ACL_CHECK(aclrtCreateEventWithFlag(&(event->Get()), ACL_EVENT_SYNC | ACL_EVENT_TIME_LINE));
}

template <>
void EventCreateWithFlagsT(EventT<DEVICE_TYPE_ASCEND>* event, unsigned int flags) {
  unsigned int acl_flags = ACL_EVENT_SYNC | ACL_EVENT_TIME_LINE;
  if (flags & ACL_EVENT_TIME_LINE) {
    acl_flags = acl_flags ^ ACL_EVENT_TIME_LINE;
  }

  ACL_CHECK(aclrtCreateEventWithFlag(&(event->Get()), acl_flags));
}

template <>
void EventDestroyT<DEVICE_TYPE_ASCEND>(EventT<DEVICE_TYPE_ASCEND> event) {
  ACL_CHECK(aclrtDestroyEvent(event.Get()));
}

template <>
void StreamSynchronizeT<DEVICE_TYPE_ASCEND>(StreamT<DEVICE_TYPE_ASCEND> stream) {
  ACL_CHECK(aclrtSynchronizeStream(stream.Get()));
}

template <>
void DeviceSynchronizeT<DEVICE_TYPE_ASCEND>() {
  ACL_CHECK(aclrtSynchronizeDevice());
}

template <>
void SetDeviceT<DEVICE_TYPE_ASCEND>(int device_id) {
  ACL_CHECK(aclrtSetDevice(device_id));
}

template <>
void GetDeviceT<DEVICE_TYPE_ASCEND>(int* device_id) {
  ACL_CHECK(aclrtGetDevice(device_id));
}

template <>
void GetDeviceCountT<DEVICE_TYPE_ASCEND>(int* count) {
  uint32_t tmp_count = 0;
  ACL_CHECK(aclrtGetDeviceCount(&tmp_count));
  *count = static_cast<int>(tmp_count);
}

template <>
void EventRecordT<DEVICE_TYPE_ASCEND>(EventT<DEVICE_TYPE_ASCEND> event, StreamT<DEVICE_TYPE_ASCEND> stream) {
  ACL_CHECK(aclrtRecordEvent(event.Get(), stream.Get()));
}

template <>
void StreamWaitEventT<DEVICE_TYPE_ASCEND>(StreamT<DEVICE_TYPE_ASCEND> stream, EventT<DEVICE_TYPE_ASCEND> event) {
  ACL_CHECK(aclrtStreamWaitEvent(stream.Get(), event.Get()));
}

template <>
void EventSynchronizeT<DEVICE_TYPE_ASCEND>(EventT<DEVICE_TYPE_ASCEND> event) {
  ACL_CHECK(aclrtSynchronizeEvent(event.Get()));
}

template <>
void EventElapsedTimeT<DEVICE_TYPE_ASCEND>(float* ms, EventT<DEVICE_TYPE_ASCEND> start,
                                           EventT<DEVICE_TYPE_ASCEND> end) {
  ACL_CHECK(aclrtEventElapsedTime(ms, start.Get(), end.Get()));
}

template <>
void MemGetInfoT<DEVICE_TYPE_ASCEND>(size_t* free, size_t* total) {
  ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, free, total));
}

template <>
void MallocT<DEVICE_TYPE_ASCEND>(void** dev_ptr, size_t size) {
  // NOTE(karlluo): 910B only have HBM
  ACL_CHECK(aclrtMalloc(dev_ptr, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
}

template <>
void FreeT<DEVICE_TYPE_ASCEND>(void* dev_ptr) {
  ACL_CHECK(aclrtFree(dev_ptr));
}

template <>
void HostAllocT<DEVICE_TYPE_ASCEND>(void** host_ptr, size_t size) {
  ACL_CHECK(aclrtMallocHost(host_ptr, size));
}

template <>
void FreeHostT<DEVICE_TYPE_ASCEND>(void* host_ptr) {
  ACL_CHECK(aclrtFreeHost(host_ptr));
}

template <>
void MallocAsyncT<DEVICE_TYPE_ASCEND>(void** dev_ptr, size_t size, StreamT<DEVICE_TYPE_ASCEND> stream) {
  // NOTE(karlluo): 910B only have HBM
  ACL_CHECK(aclrtMalloc(dev_ptr, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
}

template <>
void FreeAsyncT<DEVICE_TYPE_ASCEND>(void* dev_ptr, StreamT<DEVICE_TYPE_ASCEND> stream) {
  ACL_CHECK(aclrtFree(dev_ptr));
}

template <>
void MemsetAsyncT<DEVICE_TYPE_ASCEND>(void* dev_ptr, int value, size_t count, StreamT<DEVICE_TYPE_ASCEND> stream) {
  ACL_CHECK(aclrtMemsetAsync(dev_ptr, count, value, count, stream.Get()));
}

template <>
void MemsetT<DEVICE_TYPE_ASCEND>(void* dev_ptr, int value, size_t count) {
  ACL_CHECK(aclrtMemset(dev_ptr, count, value, count));
}

aclrtMemcpyKind GetAclMemcpyKind(enum MemcpyKind kind) {
  switch (kind) {
    case MEMCPY_HOST_TO_HOST:
      return ACL_MEMCPY_HOST_TO_HOST;
    case MEMCPY_HOST_TO_DEVICE:
      return ACL_MEMCPY_HOST_TO_DEVICE;
    case MEMCPY_DEVICE_TO_HOST:
      return ACL_MEMCPY_DEVICE_TO_HOST;
    default:
      return ACL_MEMCPY_DEVICE_TO_DEVICE;
  }
}

template <>
void MemcpyAsyncT<DEVICE_TYPE_ASCEND>(void* dst, const void* src, size_t count, enum MemcpyKind kind,
                                      StreamT<DEVICE_TYPE_ASCEND> stream) {
  ACL_CHECK(aclrtMemcpyAsync(dst, count, src, count, GetAclMemcpyKind(kind), stream.Get()));
}

template <>
void Memcpy2DT<DEVICE_TYPE_ASCEND>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                   size_t height, enum MemcpyKind kind) {
  ACL_CHECK(aclrtMemcpy2d(dst, dpitch, src, spitch, width, height, GetAclMemcpyKind(kind)));
}

template <>
void Memcpy2DAsyncT<DEVICE_TYPE_ASCEND>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                        size_t height, enum MemcpyKind kind, StreamT<DEVICE_TYPE_ASCEND> stream) {
  ACL_CHECK(aclrtMemcpy2dAsync(dst, dpitch, src, spitch, width, height, GetAclMemcpyKind(kind), stream.Get()));
}

template <>
void MemcpyT<DEVICE_TYPE_ASCEND>(void* dst, const void* src, size_t count, enum MemcpyKind kind) {
  ACL_CHECK(aclrtMemcpy(dst, count, src, count, GetAclMemcpyKind(kind)));
}

template <>
size_t GetTypeSizeT<DEVICE_TYPE_ASCEND>(DataType dtype) {
  static const std::unordered_map<DataType, size_t> type_map{
      {TYPE_BOOL, sizeof(bool)},       {TYPE_BYTES, sizeof(char)},      {TYPE_UINT8, sizeof(uint8_t)},
      {TYPE_UINT16, sizeof(uint16_t)}, {TYPE_UINT32, sizeof(uint32_t)}, {TYPE_UINT64, sizeof(uint64_t)},
      {TYPE_INT8, sizeof(int8_t)},     {TYPE_INT16, sizeof(int16_t)},   {TYPE_INT32, sizeof(int32_t)},
      {TYPE_INT64, sizeof(int64_t)},   {TYPE_FP16, sizeof(int16_t)},    {TYPE_FP32, sizeof(float)},
      {TYPE_FP64, sizeof(double)},     {TYPE_POINTER, sizeof(void*)}};
  return type_map.at(dtype);
}

template <class U>
DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl() {
  if (std::is_same<U, float>::value || std::is_same<U, const float>::value) {
    return TYPE_FP32;
  } else if (std::is_same<U, aclFloat16>::value || std::is_same<U, const aclFloat16>::value) {
    return TYPE_FP16;
  } else if (std::is_same<U, int>::value || std::is_same<U, const int>::value) {
    return TYPE_INT32;
  } else if (std::is_same<U, int8_t>::value || std::is_same<U, const int8_t>::value) {
    return TYPE_INT8;
  } else if (std::is_same<U, uint8_t>::value || std::is_same<U, const uint8_t>::value) {
    return TYPE_UINT8;
  } else if (std::is_same<U, unsigned int>::value || std::is_same<U, const unsigned int>::value) {
    return TYPE_UINT32;
  } else if (std::is_same<U, unsigned long>::value || std::is_same<U, const unsigned long>::value) {
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

template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<float>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<aclFloat16>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<int>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<int8_t>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<uint8_t>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<unsigned int>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<unsigned long>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<bool>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<char>();

}  // namespace ksana_llm
