/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/load_weight.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

Status LoadWeightResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeStatus(result_status, ptr, *bytes);
  return Status();
}

Status LoadWeightResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  result_status = DeserializeStatus(ptr, offset);
  return Status();
}

void LoadWeightCallable::SetWeightInstance(std::shared_ptr<WeightInstanceInterface> model_weight) {
  model_weight_ = model_weight;
}

Status LoadWeightCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  model_weight_->Load();
  return Status();
}

Status LoadWeightCallback::Call(const IpcResponse* response, IpcContext* context) {
  const LoadWeightResponse* rsp = reinterpret_cast<const LoadWeightResponse*>(response);
  if (!rsp->result_status.OK()) {
    KLLM_LOG_ERROR << "Load model weight error: " << rsp->result_status.GetMessage();
    load_status = rsp->result_status;
  }
  Notify();

  return Status();
}

}  // namespace ksana_llm
