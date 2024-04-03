/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <unistd.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/device_types.h"

namespace ksana_llm {

struct RoPEScalingFactor {
  std::string type{"default"};
  float factor{1.0f};
};

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

  size_t max_scheduler_token_num;

  // TODO(karlluo): Quant mode

  int tensor_para_size;

  size_t head_num;
  uint32_t size_per_head;
  uint32_t inter_size;
  uint32_t hidden_units;
  uint32_t num_layer;
  uint32_t rotary_embedding;
  float rope_theta;
  float layernorm_eps;
  uint32_t vocab_size;
  int start_id;
  int end_id;
  size_t num_key_value_heads;
  int max_batch_size;
  int max_position_embeddings;
  int block_token_num;

  RoPEScalingFactor rope_scaling_factor_config;

  // others attributes
  std::unordered_map<std::string, std::string> model_attributes;
};

struct RequestBatchingConfig {};

struct ContextCachingConfig {};

enum PreemptMode { SWAP = 0, RECOMPUTE = 1 };

struct BatchSchedulerConfig {
  // Max waiting time in millisecond.
  size_t waiting_timeout_in_ms = 600000;

  // The max queue len of waiting request.
  size_t max_waiting_queue_len = 256;

  // The max token number for one scheduler step.
  size_t max_token_number = 4096;

  // The max batch size.
  size_t max_batch_size = 8;

  // The max vocab size.
  size_t max_vocab_size = 32000;

  // The max input sequeue length.
  size_t max_input_len = 1024;

  // The max output sequeue length.
  size_t max_output_len = 1024;

  // The swapin block threshold.
  float swapout_block_threshold = 1.0;

  // The swapout block threshold.
  float swapin_block_threshold = 2.0;

  // The launch block threshold.
  float launch_block_threshold = 2.0;

  // The threadpool size used for swap in/out.
  size_t swap_threadpool_size = 8;

  // The preempt mode in case of insufficient GPU blocks.
  PreemptMode preempt_mode = SWAP;
};

struct LoraCoordinatorConfig {};

struct AllocatorConfig {
  // The preallocated blocks.
  int64_t blocks_num = 0;

  // The block size, in bytes.
  int64_t block_size;

  // The max token number of one block.
  size_t block_token_num;

  MemoryDevice device;
};

struct BlockManagerConfig {
  // The config of allocator for cpu/gpu/npu.
  AllocatorConfig host_allocator_config;
  AllocatorConfig device_allocator_config;

  // The ratio of reserved device memory.
  float reserved_device_memory_ratio = 0.05;

  // The ratio of lora device memory.
  float lora_deivce_memory_ratio = 0.0;

  // The ratio of block device memory. use all left memory if less than 0.0.
  float block_device_memory_ratio = -1.0;

  // The scale fator of lora host memory.
  float lora_host_memory_factor = 10.0;

  // The scale fator of block host memory.
  float block_host_memory_factor = 10.0;

  // Prefix cache length cache token numbers of prompt prefix
  // default is 0, disable this function
  // positve integer is the real length of this prompt prefix
  // TODO(karlluo): -1 is autofix mode
  int prefix_cache_len = 0;
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

// The config of profiler.
struct ProfilerConfig {
  // The stat interval, in second.
  size_t stat_interval_second;

  // The stat buffer size.
  size_t stat_buffer_size;

  // The async report thread num.
  size_t report_threadpool_size;
};

class Environment {
 public:
  Environment() {}

  // Parse environment from YAML config file.
  Status ParseConfig(const std::string &config_file);

  // Parse model config from model dir.
  Status ParseModelConfig(const std::string &model_name, const std::string &model_dir);

  // Parse command line options.
  Status ParseOptions(int argc, char **argv);

  // Get the model configs from env.
  Status GetModelConfigs(std::unordered_map<std::string, ModelConfig> &model_configs);

  // Get the model config by name.
  Status GetModelConfig(const std::string &model_name, ModelConfig &model_config);

  // Get the config of a batch manager.
  Status GetBatchManagerConfig(BatchManagerConfig &batch_manager_config);

  // Get the config of block manager.
  Status GetBlockManagerConfig(BlockManagerConfig &block_manager_config);

  // Get the config of endpoint.
  Status GetEndpointConfig(EndpointConfig &endpoint_config);

  // Get the config of profiler.
  Status GetProfilerConfig(ProfilerConfig &profiler_config);

  size_t GetTensorParallelSize() { return tensor_parallel_size_; }

  size_t GetPipeLineParallelSize() { return pipeline_parallel_size_; }

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

  // The config of profiler.
  ProfilerConfig profiler_config_;

  size_t tensor_parallel_size_{0};
  size_t pipeline_parallel_size_{0};

  // Whether lora is enabled.
  bool enable_lora_adapter_ = false;
};

}  // namespace ksana_llm
