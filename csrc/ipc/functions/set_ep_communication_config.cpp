/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/set_ep_communication_config.h"

#include "ipc/core/ipc_func_serialization.h"
#include "models/common/common_model.h"
#include "runtime/model_instance.h"
#include "utils/logger.h"

namespace ksana_llm {

Status SetEPCommunicationConfigRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeVector<EPCommunicationConfig>(aggregated_comm_configs, ptr, *bytes);
  return Status();
}

Status SetEPCommunicationConfigRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  DeserializeVector<EPCommunicationConfig>(aggregated_comm_configs, ptr, offset);
  return Status();
}

Status SetEPCommunicationConfigResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  // Empty response, no data to serialize
  return Status();
}

Status SetEPCommunicationConfigResponse::Deserialize(const void* ptr) {
  // Empty response, no data to deserialize
  return Status();
}

void SetEPCommunicationConfigCallable::SetModelInstance(std::shared_ptr<ModelInstance> model_instance) {
  model_instance_ = model_instance;
}

Status SetEPCommunicationConfigCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const SetEPCommunicationConfigRequest* set_request = static_cast<const SetEPCommunicationConfigRequest*>(request);

  if (!model_instance_) {
    KLLM_LOG_ERROR << "ModelInstance is null, cannot initialize expert parallel manager";
    return Status(RET_RUNTIME_FAILED, "ModelInstance is null");
  }

  // Get ExpertParallelManager from CommonModel's MoeLayerFactory
  CommonModel* common_model = dynamic_cast<CommonModel*>(model_instance_->GetBaseModel().get());
  if (!common_model) {
    KLLM_LOG_ERROR << "Failed to cast to CommonModel";
    return Status(RET_RUNTIME_FAILED, "Failed to cast to CommonModel");
  }

  auto expert_parallel_manager = common_model->layer_creation_context_.moe_layer_factory->GetExpertParallelManager();
  if (!expert_parallel_manager) {
    KLLM_LOG_ERROR << "Failed to get ExpertParallelManager from MoeLayerFactory";
    return Status(RET_RUNTIME_FAILED, "Failed to get ExpertParallelManager from MoeLayerFactory");
  }

  // Use aggregated communicator config to initialize expert parallel.
  // These configs are collected from all executors
  KLLM_LOG_INFO << "Calling ExpertParallelManager::Initialize with aggregated communicator config from all executors "
                   "aggregated_comm_configs.size()="
                << set_request->aggregated_comm_configs.size();

  // Initialize will call the strategy's Initialize method.
  // The aggregated_comm_configs contains initialization data from all executors.
  Status init_status = expert_parallel_manager->Initialize(set_request->aggregated_comm_configs);
  if (!init_status.OK()) {
    KLLM_LOG_ERROR << "Failed to initialize expert parallel manager: " << init_status.GetMessage();
    return init_status;
  }

  return Status();
}

Status SetEPCommunicationConfigCallback::Call(const IpcResponse* response, IpcContext* context) {
  Notify();
  return Status();
}

}  // namespace ksana_llm