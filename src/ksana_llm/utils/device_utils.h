/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_CUDA
#  include "ksana_llm/utils/nvidia/nvidia_device.h"
#endif

#ifdef ENABLE_ACL
#  include "ksana_llm/utils/ascend/ascend_device.h"
#endif

namespace ksana_llm {

// The stream for different device type.
using Stream = StreamT<ACTIVE_DEVICE_TYPE>;

// The event for different device type.
using Event = EventT<ACTIVE_DEVICE_TYPE>;

}  // namespace ksana_llm

// The event create & destroy.
#define EventCreate EventCreateT<ACTIVE_DEVICE_TYPE>
#define EventDestroy EventDestroyT<ACTIVE_DEVICE_TYPE>

// The event create flag
#define EVENT_DEFAULT 0x00
#define EVENT_DISABLE_TIMING 0x02
#define EventCreateWithFlags EventCreateWithFlagsT<ACTIVE_DEVICE_TYPE>

// The synchronize function.
#define StreamSynchronize StreamSynchronizeT<ACTIVE_DEVICE_TYPE>
#define DeviceSynchronize DeviceSynchronizeT<ACTIVE_DEVICE_TYPE>

// The set/get device function.
#define SetDevice SetDeviceT<ACTIVE_DEVICE_TYPE>
#define GetDevice GetDeviceT<ACTIVE_DEVICE_TYPE>

#define GetDeviceCount GetDeviceCountT<ACTIVE_DEVICE_TYPE>

// The event function.
#define EventRecord EventRecordT<ACTIVE_DEVICE_TYPE>
#define StreamWaitEvent StreamWaitEventT<ACTIVE_DEVICE_TYPE>

#define EventSynchronize EventSynchronizeT<ACTIVE_DEVICE_TYPE>
#define EventElapsedTime EventElapsedTimeT<ACTIVE_DEVICE_TYPE>

#define MemGetInfo MemGetInfoT<ACTIVE_DEVICE_TYPE>

// The malloc & free
#define Malloc MallocT<ACTIVE_DEVICE_TYPE>
#define Free FreeT<ACTIVE_DEVICE_TYPE>

// The alloc/free host memory
#define HostAlloc HostAllocT<ACTIVE_DEVICE_TYPE>
#define FreeHost FreeHostT<ACTIVE_DEVICE_TYPE>

// The malloc/free device memory
#define MallocAsync MallocAsyncT<ACTIVE_DEVICE_TYPE>
#define FreeAsync FreeAsyncT<ACTIVE_DEVICE_TYPE>

// The memset/memcpy function.
#define MemsetAsync MemsetAsyncT<ACTIVE_DEVICE_TYPE>
#define MemcpyAsync MemcpyAsyncT<ACTIVE_DEVICE_TYPE>

#define Memset MemsetT<ACTIVE_DEVICE_TYPE>
#define Memcpy MemcpyT<ACTIVE_DEVICE_TYPE>

#define Memcpy2D Memcpy2DT<ACTIVE_DEVICE_TYPE>
#define Memcpy2DAsync Memcpy2DAsyncT<ACTIVE_DEVICE_TYPE>

#define GetDataType GetDataTypeT<ACTIVE_DEVICE_TYPE>::impl

#define GetRuntimeContext GetRuntimeContextT<ACTIVE_DEVICE_TYPE>