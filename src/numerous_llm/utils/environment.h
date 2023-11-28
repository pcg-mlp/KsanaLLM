/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <string>
#include <vector>

#include "numerous_llm/utils/status.h"

namespace numerous_llm {

// The model informations.
struct ModelConfig {
  // The model name.
  std::string name;

  // The model type, such as llama.
  std::string type;

  // The dir path.
  std::string path;
};

// The config of batch manager.
struct BatchManagerConfig {};

// The config of endpoint.
struct EndpointConfig {};

class Environment {
public:
  Environment();
  ~Environment();

  // Parse command line options.
  Status ParseOptions(int argc, char **argv);

  // Get the model list from env.
  Status GetModelList(std::vector<ModelConfig> &model_configs);

  // Get the config of a batch manager.
  Status GetBatchManagerConfig(BatchManagerConfig &batch_manager_config);

  // Get the config of endpoint.
  Status GetEndpointConfig(const EndpointConfig &endpoint_config);
};

} // namespace numerous_llm
