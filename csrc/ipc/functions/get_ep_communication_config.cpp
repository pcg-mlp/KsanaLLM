/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/get_ep_communication_config.h"

#include "configure/environment.h"
#include "ipc/core/ipc_func_serialization.h"
#include "models/common/common_model.h"
#include "runtime/model_instance.h"
#include "utils/logger.h"

namespace ksana_llm {

Status GetEPCommunicationConfigRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  // Empty request, no data to serialize
  return Status();
}

Status GetEPCommunicationConfigRequest::Deserialize(const void* ptr) {
  // Empty request, no data to deserialize
  return Status();
}

Status GetEPCommunicationConfigResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializePod<EPCommunicationConfig>(executor_comm_config, ptr, *bytes);
  return Status();
}

Status GetEPCommunicationConfigResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  executor_comm_config = DeserializePod<EPCommunicationConfig>(ptr, offset);
  return Status();
}

void GetEPCommunicationConfigCallable::SetModelInstance(std::shared_ptr<ModelInstance> model_instance) {
  model_instance_ = model_instance;
}

void GetEPCommunicationConfigCallable::SetEnvironment(std::shared_ptr<Environment> env) { env_ = env; }

Status GetEPCommunicationConfigCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  GetEPCommunicationConfigResponse* get_response = static_cast<GetEPCommunicationConfigResponse*>(response);
  if (!model_instance_) {
    KLLM_LOG_ERROR << "ModelInstance is null, cannot get communicator config";
    return Status(RET_RUNTIME_FAILED, "ModelInstance is null");
  }

  if (!env_) {
    KLLM_LOG_ERROR << "Environment is null, cannot get communicator config";
    return Status(RET_RUNTIME_FAILED, "Environment is null");
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

  // Get rank from IpcContext
  int rank = context->to_rank;

  // Get expert parallel config from environment
  ExpertParallelConfig expert_parallel_config;
  Status config_status = env_->GetExpertParallelConfig(expert_parallel_config);
  if (!config_status.OK()) {
    KLLM_LOG_ERROR << "Failed to get ExpertParallelConfig: " << config_status.GetMessage();
    return config_status;
  }
  size_t expert_world_size = expert_parallel_config.expert_world_size;
  size_t expert_para_size = expert_parallel_config.expert_para_size;
  size_t expert_node_rank = expert_parallel_config.expert_node_rank;

  // Get model config from environment
  ModelConfig model_config;
  Status model_config_status = env_->GetModelConfig(model_config);
  if (!model_config_status.OK()) {
    KLLM_LOG_ERROR << "Failed to get ModelConfig: " << model_config_status.GetMessage();
    return model_config_status;
  }
  size_t num_experts = model_config.moe_config.num_experts;

  // Create ExpertParallelManager
  Status create_status = expert_parallel_manager->Create(
      rank, static_cast<int>(expert_para_size * expert_world_size), static_cast<int>(expert_world_size),
      static_cast<int>(expert_node_rank), static_cast<int>(model_config.hidden_units), static_cast<int>(num_experts));
  if (!create_status.OK()) {
    KLLM_LOG_ERROR << "Failed to create ExpertParallelManager: " << create_status.GetMessage();
    return create_status;
  }

  KLLM_LOG_DEBUG << "ExpertParallelManager created successfully for rank " << rank;

  // Get communicator config via ExpertParallelManager
  KLLM_LOG_DEBUG << "Get EP communicator config";
  get_response->executor_comm_config = expert_parallel_manager->GetCommunicationConfig();

  // Check if communicator config is valid (non-empty)
  bool is_valid = false;
  for (int i = 0; i < kNvshmemUniqueIdSize; ++i) {
    if (get_response->executor_comm_config.nvshmem_id[i] != 0) {
      is_valid = true;
      break;
    }
  }

  if (!is_valid) {
    KLLM_LOG_ERROR << "Failed to get EP communicator config: config is empty";
    return Status(RET_RUNTIME_FAILED, "Failed to get EP communicator config");
  }
  return Status();
}

Status GetEPCommunicationConfigCallback::Call(const IpcResponse* response, IpcContext* context) {
  const GetEPCommunicationConfigResponse* get_response = static_cast<const GetEPCommunicationConfigResponse*>(response);

  int executor_id = context->to_rank;
  if (executor_id >= 0 && executor_id < static_cast<int>(executor_comm_configs.size())) {
    executor_comm_configs[executor_id] = get_response->executor_comm_config;
  }

  Notify();
  return Status();
}

}  // namespace ksana_llm