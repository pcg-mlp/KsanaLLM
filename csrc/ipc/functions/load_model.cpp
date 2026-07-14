/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/load_model.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

Status LoadModelResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(result_status, ptr, *bytes);
  return Status();
}

Status LoadModelResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  result_status = DeserializeStatus(ptr, offset);
  return Status();
}

void LoadModelCallable::SetModelInstance(std::shared_ptr<ModelInstance> model_instance) {
  model_instance_ = model_instance;
}

Status LoadModelCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  model_instance_->Load();
  return Status();
}

Status LoadModelCallback::Call(const IpcResponse* response, IpcContext* context) {
  const LoadModelResponse* rsp = reinterpret_cast<const LoadModelResponse*>(response);
  if (!rsp->result_status.OK()) {
    KLLM_LOG_ERROR << "Load model instance error: " << rsp->result_status.GetMessage();
    load_status = rsp->result_status;
  }
  Notify();
  return Status();
}

}  // namespace ksana_llm
