/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/utils/environment.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "fmt/core.h"
#include "gflags/gflags.h"

#include "3rdparty/ini_reader.h"
#include "numerous_llm/utils/logger.h"
#include "numerous_llm/utils/status.h"

DEFINE_string(model_config, "./config.ini", "Get the model config file path");
DEFINE_string(host, "localhost", "HTTP service hostname, default is localhost");
DEFINE_int32(port, 8080, "HTTP service port, default is 8080");

namespace numerous_llm {

inline bool IsFileExists(const std::string &file_path) {
  std::ifstream f(file_path.c_str());
  return f.good();
}

DataType GetModelDataType(const INIReader &ini_reader) {
  std::string data_type_raw_str = ini_reader.Get("ft_instance_hyperparameter", "data_type");
  std::string unified_data_type_raw_str = data_type_raw_str;
  // unify it to lower case
  std::transform(unified_data_type_raw_str.begin(), unified_data_type_raw_str.end(), unified_data_type_raw_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (unified_data_type_raw_str == "fp16") {
    return DataType::TYPE_FP16;
  } else {
    throw std::runtime_error("Not supported model data type.");
  }
}

void PrepareModeAttirbutes(const INIReader &ini_reader, ModelConfig &model_config) {
  model_config.head_num = ini_reader.GetInteger(model_config.name, "head_num");
  model_config.num_key_value_heads =
      ini_reader.GetInteger(model_config.name, "num_key_value_heads", model_config.head_num);
  model_config.size_per_head = ini_reader.GetInteger(model_config.name, "size_per_head");
  model_config.inter_size = ini_reader.GetInteger(model_config.name, "inter_size");
  model_config.vocab_size = ini_reader.GetInteger(model_config.name, "vocab_size");
  model_config.num_layer = ini_reader.GetInteger(model_config.name, "num_layer");
  model_config.rotary_embedding = ini_reader.GetInteger(model_config.name, "rotary_embedding");
  model_config.rope_theta = ini_reader.GetFloat(model_config.name, "rope_theta", 10000.0f);
  model_config.layernorm_eps = ini_reader.GetFloat(model_config.name, "layernorm_eps");
  model_config.start_id = ini_reader.GetInteger(model_config.name, "start_id");
  model_config.end_id = ini_reader.GetInteger(model_config.name, "end_id");
}

Status Environment::ParseOptions(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!IsFileExists(FLAGS_model_config)) {
    NLLM_LOG_ERROR << fmt::format("Model config file: {} is not exists.", FLAGS_model_config) << std::endl;
    return Status(RetCode::RET_SEGMENT_FAULT);
  }

  INIReader ini_reader = INIReader(FLAGS_model_config);
  if (ini_reader.ParseError() < 0) {
    NLLM_LOG_ERROR << fmt::format("Load model config file: {} error.", FLAGS_model_config) << std::endl;
    return Status(RetCode::RET_SEGMENT_FAULT);
  }

  tensor_parallel_size_ = ini_reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size");
  pipeline_parallel_size_ = ini_reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size");

  if (!(pipeline_parallel_size_ > 0 && tensor_parallel_size_ > 0)) {
    throw std::runtime_error("tensor_para_size and pipeline_para_size should > 0");
  }

  ModelConfig model_config;
  model_config.name = ini_reader.Get("ft_instance_hyperparameter", "model_name");
  model_config.path = ini_reader.Get("ft_instance_hyperparameter", "model_dir");
  model_config.weight_data_type = GetModelDataType(ini_reader);
  model_config.tensor_para_size = tensor_parallel_size_;
  PrepareModeAttirbutes(ini_reader, model_config);

  // TODO: Get from config.
  model_config.max_token_num = 1024;

  model_configs_.push_back(model_config);

  NLLM_LOG_INFO << fmt::format("Load model {} from config file: {} success.", model_config.name, model_config.path);

  endpoint_config_.host = FLAGS_host;
  endpoint_config_.port = static_cast<uint32_t>(FLAGS_port);

  endpoint_config_.host = FLAGS_host;
  endpoint_config_.port = static_cast<uint32_t>(FLAGS_port);

  InitializeBlockManagerConfig();

  return Status();
}

void Environment::InitializeBlockManagerConfig() {
  NLLM_CHECK_WITH_INFO(model_configs_.size() > 0, "No model configed.");
  const ModelConfig &model_config = model_configs_.front();

  block_manager_config_.cpu_allocator_config.block_token_num = block_token_num_;
  block_manager_config_.device_allocator_config.block_token_num = block_token_num_;

  size_t token_size = (model_config.num_layer / GetPipeLineParallelSize()) *
                      (model_config.head_num / GetTensorParallelSize()) * model_config.size_per_head;

  block_manager_config_.cpu_allocator_config.block_size = token_size * block_token_num_;
  block_manager_config_.device_allocator_config.block_size = token_size * block_token_num_;

  block_manager_config_.cpu_allocator_config.device = MemoryDevice::MEMORY_CPU;
  block_manager_config_.device_allocator_config.device = MemoryDevice::MEMORY_GPU;

  // TODO(yancyliu): should calculated through device memory useage.
  block_manager_config_.cpu_allocator_config.blocks_num = 128;
  block_manager_config_.device_allocator_config.blocks_num = 128;
}

Status Environment::GetModelList(std::vector<ModelConfig> &model_configs) {
  model_configs = model_configs_;
  return Status();
}

Status Environment::GetBatchManagerConfig(BatchManagerConfig &batch_manager_config) {
  batch_manager_config = batch_manager_config_;
  return Status();
}

Status Environment::GetBlockManagerConfig(BlockManagerConfig &block_manager_config) {
  block_manager_config = block_manager_config_;
  return Status();
}

Status Environment::GetEndpointConfig(EndpointConfig &endpoint_config) {
  endpoint_config = endpoint_config_;
  return Status();
}

}  // namespace numerous_llm
