/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "httplib.h"

#include "numerous_llm/block_manager/memory_block.h"
#include "numerous_llm/utils/dtypes.h"
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

  // Type of weight
  DataType weight_data_type;

  // The max number of (input + output tokens)
  size_t max_token_num;

  // TODO(karlluo): Quant mode

  // Device Type
  MemoryDevice memory_device;
  int tensor_para_size;

  size_t head_num;
  uint32_t size_per_head;
  uint32_t inter_size;
  uint32_t num_layer;
  uint32_t rotary_embedding;
  float rope_theta;
  float layernorm_eps;
  uint32_t vocab_size;
  int start_id;
  int end_id;
  size_t num_key_value_heads;
  int default_batch_size;
  int max_position_embeddings;

  // others attributes
  std::unordered_map<std::string, std::string> model_attributes;
};

struct RequestBatchingConfig {};

struct ContextCachingConfig {};

struct BatchSchedulerConfig {
  // Max waiting time in millisecond.
  size_t timeout_in_ms = 60000;

  // The max queue len of waiting request.
  size_t max_waiting_queue_len = 100;

  // The max token number for one scheduler step.
  size_t max_token_number = 4096;
};

struct LoraCoordinatorConfig {};

struct AllocatorConfig {
  // The preallocated blocks.
  int64_t blocks_num;

  // The block size, in bytes.
  int64_t block_size;

  // The max token number of one block.
  size_t block_token_num;

  MemoryDevice device;
};

struct BlockManagerConfig {
  // The config of allocator for cpu/gpu/npu.
  AllocatorConfig cpu_allocator_config;
  AllocatorConfig device_allocator_config;
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
};

// The endpoint type.
enum EndpointType { ENDPOINT_LOCAL, ENDPOINT_HTTP, ENDPOINT_TRPC };

// The config of endpoint.
struct EndpointConfig {
  // The endpoint type.
  EndpointType type = EndpointType::ENDPOINT_LOCAL;

  // HTTP service hostname, default is localhost
  std::string host;

  // HTTP service port, default is 8080
  uint32_t port;
};

class Environment {
 public:
  // Parse environment from config file.
  Status ParseConfig(const std::string &config_file);

  // Parse command line options.
  Status ParseOptions(int argc, char **argv);

  // Get the model configs from env.
  Status GetModelConfigs(std::unordered_map<std::string, ModelConfig> &model_configs);

  // Get the model config by name.
  Status GetModelConfig(const std::string& model_name, ModelConfig& model_config);

  // Get the config of a batch manager.
  Status GetBatchManagerConfig(BatchManagerConfig &batch_manager_config);

  // Get the config of block manager.
  Status GetBlockManagerConfig(BlockManagerConfig &block_manager_config);

  // Get the config of endpoint.
  Status GetEndpointConfig(EndpointConfig &endpoint_config);

  int GetTensorParallelSize() { return tensor_parallel_size_; }

  int GetPipeLineParallelSize() { return pipeline_parallel_size_; }

 private:
  // Calculate block size via model configs.
  void InitializeBlockManagerConfig();

  // Check Whether the environment config is valid.
  Status CheckEnvironment();

 private:
  // The model list that should be loaded.
  std::unordered_map<std::string, ModelConfig> model_configs_;

  // The config of batch manager.
  BatchManagerConfig batch_manager_config_;

  // The config of block manager.
  BlockManagerConfig block_manager_config_;

  // The config of endpoint.
  EndpointConfig endpoint_config_;

  int tensor_parallel_size_{0};
  int pipeline_parallel_size_{0};

  // The max token number of one block.
  size_t block_token_num_ = 16;
};

}  // namespace numerous_llm
