/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/utils/environment.h"

namespace numerous_llm {

Status Environment::ParseOptions(int argc, char **argv) {
  ModelConfig model_config;
  model_config.name = "llama-13b";
  model_config.type = "llama";
  model_config.path = ".";
  model_configs_.push_back(model_config);

  return Status();
}

Status Environment::GetModelList(std::vector<ModelConfig> &model_configs) {
  model_configs = model_configs_;
  return Status();
}

Status
Environment::GetBatchManagerConfig(BatchManagerConfig &batch_manager_config) {
  batch_manager_config = batch_manager_config_;
  return Status();
}

Status Environment::GetEndpointConfig(EndpointConfig &endpoint_config) {
  endpoint_config = endpoint_config_;
  return Status();
}

} // namespace numerous_llm
