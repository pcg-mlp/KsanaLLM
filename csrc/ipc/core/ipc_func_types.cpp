/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/core/ipc_func_types.h"

#include "ipc/core/ipc_func_serialization.h"

namespace ksana_llm {

Status IpcRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  return Status();
}

Status IpcRequest::Deserialize(const void* ptr) { return Status(); }

Status IpcResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  return Status();
}

Status IpcResponse::Deserialize(const void* ptr) { return Status(); }

Status IpcCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) { return Status(); }

Status IpcCallback::Call(const IpcResponse* response, IpcContext* context) { return Status(); }

}  // namespace ksana_llm
