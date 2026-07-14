/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/add_executor.h"
#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

Status AddExecutorRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializePod<size_t>(executor_pid, ptr, *bytes);
  return Status();
}

Status AddExecutorRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  executor_pid = DeserializePod<size_t>(ptr, offset);
  return Status();
}

Status AddExecutorCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const AddExecutorRequest* real_req = reinterpret_cast<const AddExecutorRequest*>(request);
  executor_pids.push_back(real_req->executor_pid);

  Notify();
  return Status();
}

}  // namespace ksana_llm
