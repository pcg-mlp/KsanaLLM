/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "configure/model_config_parser.h"
#include "configure/schedule_config_parser.h"
#include "configure/yaml_reader.h"
#include "connector/config.h"
#include "device/device_types.h"
#include "global_cache_connector/global_cache_connector_types.h"
#include "utils/logger.h"
#include "utils/search_path.h"
#include "utils/status.h"

namespace ksana_llm {

constexpr size_t kStepGenerateTokenNum = 1;  // The number of tokens that the model generated at each step

// Threshold for determining decode vs prefill requests
// `input_ids_len <= decode_token_num_threshold` will regard as decode (use page attention), default is 1
size_t GetDecodeTokenNumThreshold();
// Vocabulary information for structured output generation
struct VocabInfo {
  std::vector<std::string> vocab;
  int vocab_size = 0;               // Vocabulary size (int is sufficient: max ~2B >> actual ~500K)
  std::vector<int> stop_token_ids;  // EOS token IDs from tokenizer
  bool initialized = false;

  // Tokenizer type information (to avoid Python API calls in Executor process)
  int vocab_type = 0;  // 0: RAW, 1: BYTE_FALLBACK, 2: BYTE_LEVEL
  bool add_prefix_space = false;

  // Reasoning configuration
  int think_end_token_id = -1;  // Token ID that marks the end of reasoning phase
};

// The endpoint type.
enum class EndpointType { LOCAL, RPC };

// The config of endpoint.
struct EndpointConfig {
  // The endpoint service type.
  EndpointType type = EndpointType::LOCAL;

  // If the endpoint type is RPC, load the corresponding
  // shared library based on the rpc plugin name.
  std::string rpc_plugin_name;

  // The endpoint service host address.
  std::string host = "0.0.0.0";

  // The endpoint service port.
  uint32_t port = 8080;

  // Whether to enable the endpoint access log.
  bool access_log = true;
};

// The config of profiler.
struct ProfilerConfig {
  // The stat interval, in second.
  std::string metrics_export_url;
  uint64_t export_interval_millis;
  uint64_t export_timeout_millis;

  // Opentelemetry Resource attributes.
  std::unordered_map<std::string, std::string> resource_attributes;
};

class Environment {
 public:
  Environment();

  void InitializeAttentionBackend();

  // Parse environment from YAML config file.
  // model_dir_override is used to override model_dir and tokenizer_dir in config_file
  Status ParseConfig(const std::string &config_file, const std::string &model_dir_override = "",
                     const std::string &model_config_filename = "config.json");

  // Get the model config
  Status GetModelConfig(ModelConfig &model_config);

  // Get the config of batch manager.
  Status GetBatchSchedulerConfig(BatchSchedulerConfig &batch_manager_config);

  // Get the config of cached manager.
  Status GetCacheManagerConfig(CacheManagerConfig &cache_manager_config);

  Status GetRuntimeConfig(RuntimeConfig &runtime_config);

  size_t GetTransferLayerChunkSize();

  // Get the config of block manager.
  Status GetBlockManagerConfig(BlockManagerConfig &block_manager_config);

  // TODO(yancyliu): remove from here later.
  Status CalculateBlockNumber(size_t device_free, size_t deivce_total);
  Status ResetPipelineBlockNumber();
  size_t GetTotalDeviceBlockNum();
  size_t GetTotalHostBlockNum();
  std::vector<int> GetDataParaGroupDevices(int dp_id);

  // Get the config of profiler.
  Status GetProfilerConfig(ProfilerConfig &profiler_config);

  const std::string &GetYamlGptqBackend() const { return yaml_gptq_backend_; }

  const std::string &GetYamlWeightQuantMethod() const { return yaml_weight_quant_method_; }

  bool IsReportVersion() { return is_version_report_; }

  size_t GetMaxBatchSize() const { return schedule_config_parser_.GetMaxBatchSize(); }

  Status GetPipelineConfig(PipelineConfig &pipeline_config) const {
    return schedule_config_parser_.GetPipelineConfig(pipeline_config);
  }

  Status GetExpertParallelConfig(ExpertParallelConfig &expert_parallel_config) const {
    return schedule_config_parser_.GetExpertParallelConfig(expert_parallel_config);
  }

  Status GetConnectorConfigs(ConnectorConfig &connector_config) const {
    return schedule_config_parser_.GetConnectorConfigs(connector_config);
  }

  // Calculate block size via model configs.
  Status InitializeBlockManagerConfig() { return schedule_config_parser_.InitializeBlockManagerConfig(model_config_); }

  // Init Expert-Parallel Config from env.
  void InitializeExpertParallelConfig() { schedule_config_parser_.InitializeExpertParallelConfig(); }

 public:  // Used for test.
  // Update model config after batch scheduler config is updated.
  // TODO(robertyuan): remove later.
  Status UpdateModelConfig();

  void SetModelConfig(ModelConfig &model_config) { model_config_ = model_config; }
  void SetBatchSchedulerConfig(BatchSchedulerConfig &batch_manager_config) {
    schedule_config_parser_.SetBatchSchedulerConfig(batch_manager_config);
  }
  void SetCacheManagerConfig(CacheManagerConfig &cache_manager_config) {
    schedule_config_parser_.SetCacheManagerConfig(cache_manager_config);
  }

  void SetTensorParallelSize(size_t tensor_parallel_size) {
    schedule_config_parser_.SetTensorParallelSize(tensor_parallel_size);
  }

  void SetExpertParallelConfig(const ExpertParallelConfig &expert_parallel_config) {
    return schedule_config_parser_.SetExpertParallelConfig(expert_parallel_config);
  }

  void GetAttnBackendConfig(AttnBackendConfig &attn_backend_config) {
    schedule_config_parser_.GetAttnBackendConfig(attn_backend_config);
  }
  void SetAttnBackendConfig(const AttnBackendConfig &attn_backend_config) {
    schedule_config_parser_.SetAttnBackendConfig(attn_backend_config);
  }

  void SetConnectorConfigs(const ConnectorConfig &connector_config) {
    schedule_config_parser_.SetConnectorConfigs(connector_config);
  }

  Status GetGlobalCacheConnectorConfig(GlobalCacheConnectorConfig &config) const {
    return schedule_config_parser_.GetGlobalCacheConnectorConfig(config);
  }

  void SetGlobalCacheConnectorConfig(const GlobalCacheConnectorConfig &config) {
    schedule_config_parser_.SetGlobalCacheConnectorConfig(config);
  }

  // Modify reserved_device_memory_ratio
  void SetReservedDeviceRatio(float reserved_device_memory_ratio);

  // Set and get multiple node pipeline config.
  void SetPipelineConfig(const PipelineConfig &pipeline_config) {
    return schedule_config_parser_.SetPipelineConfig(pipeline_config);
  }

  void SetAttnDataParallelSize(size_t attn_data_parallel_size) {
    schedule_config_parser_.SetAttnDataParallelSize(attn_data_parallel_size);
  }

  void SetExpertParallelSize(size_t expert_parallel_size) {
    schedule_config_parser_.SetExpertParallelSize(expert_parallel_size);
  }

  void SetBlockManagerConfig(const BlockManagerConfig &block_manager_config);

  // Get and set vocab info for structured output
  Status GetVocabInfo(VocabInfo &vocab_info) const {
    if (!vocab_info_.initialized) {
      return Status(RET_RUNTIME_FAILED, "Vocab info not initialized");
    }
    vocab_info = vocab_info_;
    return Status();
  }

  // Get vocab info by const reference (no copy, for read-only access)
  const VocabInfo &GetVocabInfoRef() const { return vocab_info_; }

  void SetVocabInfo(const VocabInfo &vocab_info) { vocab_info_ = vocab_info; }

  // Set think_end_token_id for reasoning mode
  void SetThinkEndTokenId(int think_end_token_id) {
    vocab_info_.think_end_token_id = think_end_token_id;
    KLLM_LOG_INFO << "Set think_end_token_id to " << think_end_token_id << " in Environment";
  }

  // Get think_end_token_id
  int GetThinkEndTokenId() const { return vocab_info_.think_end_token_id; }

  // Get vocab size from model config
  size_t GetVocabSize() const { return model_config_.vocab_size; }

 private:
  void Reset();

  // Parse model config from model dir.
  Status ParseModelAndScheduleConfig(YamlReader &yaml_reader, const std::string &model_dir,
                                     const std::string &tokenizer_dir, const std::string &model_config_filename);

 private:
  bool model_config_initialized_;
  ModelConfig model_config_;

  // The backend of gptq/awq quantization.
  std::string yaml_gptq_backend_;

  // The config of quantization.
  std::string yaml_weight_quant_method_;

  // The config of profiler.
  ProfilerConfig profiler_config_;

  bool is_version_report_ = true;

  ScheduleConfigParser schedule_config_parser_;

  // Vocab information for structured output (shared between Engine and Executor)
  VocabInfo vocab_info_;
};

}  // namespace ksana_llm
