/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/reset_pipeline_config.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

Status ResetPipelineConfigRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializeString(pipeline_config.master_host, ptr, *bytes);
  SerializePod<uint16_t>(pipeline_config.master_port, ptr, *bytes);

  SerializePod<size_t>(pipeline_config.world_size, ptr, *bytes);
  SerializePod<size_t>(pipeline_config.node_rank, ptr, *bytes);

  SerializePod<int16_t>(pipeline_config.lower_layer_idx, ptr, *bytes);
  SerializePod<int16_t>(pipeline_config.upper_layer_idx, ptr, *bytes);

  SerializePod<int16_t>(pipeline_config.lower_nextn_layer_idx, ptr, *bytes);
  SerializePod<uint16_t>(pipeline_config.upper_nextn_layer_idx, ptr, *bytes);

  SerializePod<size_t>(pipeline_config.device_block_num, ptr, *bytes);
  SerializePod<size_t>(pipeline_config.host_block_num, ptr, *bytes);

  SerializeString(pipeline_config.data_host, ptr, *bytes);
  SerializePod<uint16_t>(pipeline_config.data_port, ptr, *bytes);

  SerializeString(pipeline_config.downstream_host, ptr, *bytes);
  SerializePod<uint16_t>(pipeline_config.downstream_port, ptr, *bytes);

  SerializeArray<char>(pipeline_config.nccl_unique_id, 128, ptr, *bytes);
  SerializeEnum<DistributedCommunicationType>(pipeline_config.pipeline_para_comm_type, ptr, *bytes);

  return Status();
}

Status ResetPipelineConfigRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  pipeline_config.master_host = DeserializeString(ptr, offset);
  pipeline_config.master_port = DeserializePod<uint16_t>(ptr, offset);

  pipeline_config.world_size = DeserializePod<size_t>(ptr, offset);
  pipeline_config.node_rank = DeserializePod<size_t>(ptr, offset);

  pipeline_config.lower_layer_idx = DeserializePod<int16_t>(ptr, offset);
  pipeline_config.upper_layer_idx = DeserializePod<int16_t>(ptr, offset);

  pipeline_config.lower_nextn_layer_idx = DeserializePod<int16_t>(ptr, offset);
  pipeline_config.upper_nextn_layer_idx = DeserializePod<uint16_t>(ptr, offset);

  pipeline_config.device_block_num = DeserializePod<size_t>(ptr, offset);
  pipeline_config.host_block_num = DeserializePod<size_t>(ptr, offset);

  pipeline_config.data_host = DeserializeString(ptr, offset);
  pipeline_config.data_port = DeserializePod<uint16_t>(ptr, offset);

  pipeline_config.downstream_host = DeserializeString(ptr, offset);
  pipeline_config.downstream_port = DeserializePod<uint16_t>(ptr, offset);

  DeserializeArray<char>(pipeline_config.nccl_unique_id, ptr, offset);
  pipeline_config.pipeline_para_comm_type = DeserializeEnum<DistributedCommunicationType>(ptr, offset);

  return Status();
}

void ResetPipelineConfigCallable::SetEnvironment(std::shared_ptr<Environment> env) { env_ = env; }

void ResetPipelineConfigCallable::SetBlockCacheLayout(std::shared_ptr<BaseModelCacheLayout> block_cache_layout) {
  block_cache_layout_ = block_cache_layout;
}

Status ResetPipelineConfigCallable::Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) {
  const ResetPipelineConfigRequest* real_req = reinterpret_cast<const ResetPipelineConfigRequest*>(request);
  env_->SetPipelineConfig(real_req->pipeline_config);

  // Create cache memory layout.
  ModelConfig model_config;
  STATUS_CHECK_RETURN(env_->GetModelConfig(model_config));

  RuntimeConfig runtime_config;
  STATUS_CHECK_RETURN(env_->GetRuntimeConfig(runtime_config));

  PipelineConfig pipeline_config;
  STATUS_CHECK_RETURN(env_->GetPipelineConfig(pipeline_config));

  block_cache_layout_->Initialize(model_config, pipeline_config, runtime_config);

  // Get block size, must invoked before load model.
  STATUS_CHECK_RETURN(env_->InitializeBlockManagerConfig(block_cache_layout_));

  return Status();
}

Status ResetPipelineConfigCallback::Call(const IpcResponse* response, IpcContext* context) {
  Notify();
  return Status();
}

}  // namespace ksana_llm
