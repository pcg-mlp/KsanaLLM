/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/get_device_memory.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/memory_utils.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

Status GetDeviceMemoryResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializePod<size_t>(device_free, ptr, *bytes);
  SerializePod<size_t>(device_total, ptr, *bytes);
  return Status();
}

Status GetDeviceMemoryResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  device_free = DeserializePod<size_t>(ptr, offset);
  device_total = DeserializePod<size_t>(ptr, offset);
  return Status();
}

Status GetDeviceMemoryCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  GetDeviceMemoryResponse* real_rsp = reinterpret_cast<GetDeviceMemoryResponse*>(response);
  return GetDeviceMemoryInfo(MemoryDevice::MEMORY_DEVICE, &real_rsp->device_free, &real_rsp->device_total);
}

}  // namespace ksana_llm
