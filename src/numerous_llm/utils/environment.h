/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <string>
#include <vector>

#include "numerous_llm/utils/status.h"
#include "httplib.h"

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

struct RequestBatchingConfig {};

struct ContextCachingConfig {};

struct BatchSchedulerConfig {};

struct LoraCoordinatorConfig {};

struct AllocatorConfig {};

struct BlockManagerConfig {
  // The config of allocator for cpu/gpu/npu.
  AllocatorConfig allocator_config;
};

// The config of batch manager.
struct BatchManagerConfig {
  // The config of request batching.
  RequestBatchingConfig request_batching_config;

  // The config of context cache.
  ContextCachingConfig context_caching_config;

  // The config of batch schedule.
  BatchSchedulerConfig batch_scheduler_config;

  // The config of multi lora.
  LoraCoordinatorConfig lora_coordinator_config;

  // The config of block manager
  BlockManagerConfig block_manager_config;
};

// The config of endpoint.
struct EndpointConfig {
  // HTTP service hostname, default is localhost
  std::string host;

  // HTTP service port, default is 8080
  uint32_t port;
};

class Environment {
public:
  // Parse command line options.
  Status ParseOptions(int argc, char **argv);

  // Get the model list from env.
  Status GetModelList(std::vector<ModelConfig> &model_configs);

  // Get the config of a batch manager.
  Status GetBatchManagerConfig(BatchManagerConfig &batch_manager_config);

  // Get the config of endpoint.
  Status GetEndpointConfig(EndpointConfig &endpoint_config);

private:
  // The model list that should be loaded.
  std::vector<ModelConfig> model_configs_;

  // The config of batch manager.
  BatchManagerConfig batch_manager_config_;

  // The config of endpoint.
  EndpointConfig endpoint_config_;
};

} // namespace numerous_llm
