/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <unistd.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

struct RoPEScalingFactor {
  std::string type{"default"};
  float factor{1.0f};
  float low_freq_factor{1.0f};
  float high_freq_factor{4.0f};
  int original_max_position_embeddings{8192};
};

enum QuantMode { QUANT_NONE, QUANT_GPTQ, QUANT_AWQ, QUANT_FP8_E4M3 };

enum GroupQuantBackend { CUTLASS_BACKEND, MARLIN_BACKEND };

// The Quant informations.
struct QuantConfig {
  // The quant method
  QuantMode method = QUANT_NONE;

  // (gptq/awq) The quant bits
  size_t bits = 4;

  // (gptq/awq) The quant group size
  size_t group_size = 128;

  // (gptq) The desc act mode
  bool desc_act = false;

  GroupQuantBackend backend = CUTLASS_BACKEND;

  // (fp8) Whether weight is quantized in checkpoint.
  bool is_checkpoint_fp8_serialized = false;

  // (fp8) Whether input_scale is in checkpoint.
  bool is_activation_scheme_static = false;
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
  std::vector<int> end_ids;
  int pad_id;
  size_t num_key_value_heads;
  int max_batch_size;
  int max_position_embeddings;
  size_t block_token_num;
  std::vector<float> k_scales;
  std::vector<float> v_scales;

  RoPEScalingFactor rope_scaling_factor_config;

  bool tie_word_embeddings;
  bool exist_tie_embeddings_param = true;

  // The activation function used.
  std::string activation_function{"swiglu"};

  // Determines if the model is a visual llm model.
  bool is_visual;

  // Determines if the model is a quant model.
  bool is_quant;
  QuantConfig quant_config;

  // others attributes
  std::unordered_map<std::string, std::string> model_attributes;
};

enum PreemptMode { SWAP = 0, RECOMPUTE = 1 };

enum ScheduleStrategy { CONTINUOUS_BATCHING = 0, AUTO_BATCHING = 1 };

struct BatchSchedulerConfig {
  // The batch schedule strategy.
  ScheduleStrategy schedule_strategy = ScheduleStrategy::CONTINUOUS_BATCHING;

  // Max waiting time in millisecond.
  size_t waiting_timeout_in_ms = 600000;

  // The max queue len of waiting request.
  size_t max_waiting_queue_len = 256;

  // The max token number for one scheduler step.
  size_t max_step_tokens = 4096;

  // The max batch size.
  size_t max_batch_size = 8;

  // The max vocab size.
  size_t max_vocab_size = 32000;

  // The maximum length the generated tokens can have
  // orresponds to the length of the input prompt + max_new_tokens.
  size_t max_token_len = 2048;

  // The swapin block threshold.
  float swapout_block_threshold = 1.0;

  // The swapout block threshold.
  float swapin_block_threshold = 2.0;

  // The launch block threshold.
  float launch_block_threshold = 2.0;

  // The preempt mode in case of insufficient GPU blocks.
  PreemptMode preempt_mode = SWAP;

  // This parameter controls the maximum number of tokens processed in a single inference round.
  // Setting it to 256 means that during inference, each processing step (or "split") will handle up to 256 tokens.
  // If set to 0, it indicates that there is no limit on the number of tokens processed, and the model will attempt to
  // process the entire input at once. Adjusting this parameter can help balance inference speed and resource
  // consumption, especially when dealing with long texts.
  size_t split_fuse_token_num = 0;
};

struct AllocatorConfig {
  // The preallocated blocks.
  size_t blocks_num = 0;

  // The block size, in bytes.
  size_t block_size;

  // kv_cache storage type
  DataType kv_cache_dtype;

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
};

// For cached manager, used for auto-prefix-caching.
struct CacheManagerConfig {
  // The token number of every block, not changed after created.
  size_t block_token_num = 16;

  // The tp num, cache manager use this to allocat blocks for every token.
  size_t tensor_para_size = 2;

  // The thread number used for async swap in/out.
  size_t swap_threadpool_size = 2;

  // Whether enable prefix caching.
  bool enable_prefix_caching = false;
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
  std::string trace_export_url;
  std::string metrics_export_url;
  uint64_t export_interval_millis;
  uint64_t export_timeout_millis;

  // Opentelemetry Resource attributes.
  std::unordered_map<std::string, std::string> resource_attributes;
};

class Environment {
 public:
  Environment() {}

  // Parse environment from YAML config file.
  Status ParseConfig(const std::string &config_file);

  // Parse model config from model dir.
  Status ParseModelConfig(const std::string &model_dir);

  // Parse command line options.
  Status ParseOptions(int argc, char **argv);

  // Get the model configs from env.
  Status GetModelConfigs(std::unordered_map<std::string, ModelConfig> &model_configs);

  // Get the model config by name.
  Status GetModelConfig(const std::string &model_name, ModelConfig &model_config);

  // Get the config of batch manager.
  Status GetBatchSchedulerConfig(BatchSchedulerConfig &batch_manager_config);

  // Get the config of cached manager.
  Status GetCacheManagerConfig(CacheManagerConfig &cache_manager_config);

  // Whether the auto-prefix-caching is enabled.
  bool IsPrefixCachingEnabled();

  // Get the config of block manager.
  Status GetBlockManagerConfig(BlockManagerConfig &block_manager_config);

  // Get the config of endpoint.
  Status GetEndpointConfig(EndpointConfig &endpoint_config);

  // Get the config of profiler.
  Status GetProfilerConfig(ProfilerConfig &profiler_config);

  size_t GetTensorParallelSize() { return tensor_parallel_size_; }

  size_t GetPipeLineParallelSize() { return pipeline_parallel_size_; }

  bool EmbedTokensUseCpu() { return embed_tokens_use_cpu_; }

  bool IsReportVersion() { return is_version_report_; }

 private:
  // Calculate block size via model configs.
  void InitializeBlockManagerConfig();

  // Check Whether the environment config is valid.
  Status CheckEnvironment();

 private:
  // The model list that should be loaded.
  std::unordered_map<std::string, ModelConfig> model_configs_;

  // The config of batch schedule.
  BatchSchedulerConfig batch_scheduler_config_;

  // The config used by cache manager.
  CacheManagerConfig cache_manager_config_;

  // The config of block manager.
  BlockManagerConfig block_manager_config_;

  // The backend of gptq/awq quantization.
  std::string yaml_gptq_backend_;

  // The config of quantization.
  std::string yaml_weight_quant_method_;

  // The config of endpoint.
  EndpointConfig endpoint_config_;

  // The config of profiler.
  ProfilerConfig profiler_config_;

  size_t tensor_parallel_size_{0};
  size_t pipeline_parallel_size_{0};

  // Whether lora is enabled.
  bool enable_lora_adapter_ = false;

  // Embed_tokens gather operation is processed on the CPU.
  bool embed_tokens_use_cpu_ = false;
  bool is_version_report_ = true;
};

}  // namespace ksana_llm
