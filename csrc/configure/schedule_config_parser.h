/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once
#ifdef ENABLE_CUDA
#  include <nccl.h>
#endif

#include <string>
#include <unordered_map>
#include <vector>

#include "configure/model_config_parser.h"
#include "configure/yaml_reader.h"
#include "connector/config.h"
#include "device/device_types.h"
#include "global_cache_connector/global_cache_connector_types.h"
#include "pd_v2/configure/pd_v2_runtime_config.h"
#include "utils/logger.h"
#include "utils/search_path.h"
#include "utils/status.h"

namespace ksana_llm {

// Forward declare.
class BaseModelCacheLayout;

enum ScheduleStrategy { CONTINUOUS_BATCHING = 0 };

enum class SchedulerType { DEFAULT = 0, EVENT_DRIVEN = 1 };

enum PPMultibatchWBStrategy { NO_WB = 0, NO_DYNAMIC_WB = 1, WB_BATCH_REQ = 2, WB_BATCH_TOKEN = 3, WB_REQ_TOKEN = 4 };

// Configuration for Attention Data Parallel Group Balance strategy
struct AttentionDPGroupBalanceConfig {
  // Whether to enable ADP balance strategy
  bool enable_balance = false;
  // Maximum wait steps before timeout
  size_t max_waiting_steps = 50;
  // Maximum wait time in milliseconds before timeout
  size_t max_waiting_time_in_ms = 1000;
  // Minimum QPS threshold for waiting, when exceeded, set max_waiting_steps=0 and max_waiting_time_in_ms=0 (no
  // accumulation) Default -1 means all QPS will use accumulation
  double min_qps_for_waiting = -1.0;
};

struct BatchSchedulerConfig {
  // Scheduler type.
  SchedulerType scheduler_type = SchedulerType::DEFAULT;

  // The batch schedule strategy.
  ScheduleStrategy schedule_strategy = ScheduleStrategy::CONTINUOUS_BATCHING;

  // PP Multibatch workload balance strategy
  PPMultibatchWBStrategy pp_multibatch_wb_strategy = PPMultibatchWBStrategy::NO_WB;

  // Max waiting time in millisecond.
  size_t waiting_timeout_in_ms = 600000;

  // The max queue len of waiting request.
  size_t max_waiting_queue_len = 256;

  // The max token number for one scheduler step.
  size_t max_step_token_num = 4096;  // to be removed

  // The max batch size.
  size_t max_batch_size = 8;  // to be removed

  size_t max_pp_batch_num = 1;  // to be removed

  // The max vocab size.
  size_t max_vocab_size = 32000;  // TODO(robertyuan): Use model_config.vocab_size. To be removed

  // The maximum total sequence length for a request.
  // Corresponds to prompt length + generated length (not generated-only length).
  size_t max_token_len = 2048;  // to be removed

  // The launch block threshold.
  float launch_block_threshold = 2.0;

  // This parameter controls the maximum number of tokens processed in a single
  // inference round. Setting it to 256 means that during inference, each
  // processing step (or "split") will handle up to 256 tokens. If set to 0, it
  // indicates that there is no limit on the number of tokens processed, and the
  // model will attempt to process the entire input at once. Adjusting this
  // parameter can help balance inference speed and resource consumption,
  // especially when dealing with long texts.
  size_t split_fuse_token_num = 0;

  // The number of tokens per request requiring decode computation defaults to 1. When MTP or speculative decoding is
  // enabled, a single request will compute multiple tokens.
  size_t max_decode_tokens_per_req = 1;

  // The max batch size for pre-transfer operations.
  size_t max_pretransfer_batch_size = 64;

  // The number of layers to pack together for each transfer operation (chunk transfer).
  size_t transfer_layer_chunk_size = 1;

  bool enable_speculative_decoding = false;
  std::string speculative_method;

  size_t mtp_step_num = 0;

  // The token length for Parallel Token Prediction
  size_t ptp_step_num = 0;

  // The token id for Parallel Token Prediction
  uint32_t ptp_token_id = 0;

  bool enable_async = false;

  // 是否启用 Executor 端 build-ahead + device 侧 token 替换（默认关闭）。
  // 仅在 enable_async=true 且 FastPathController 启动期硬条件全部满足时真正启用；
  // 否则透明降级到原 enable_async 深度 1 路径。
  bool enable_executor_build_ahead = false;

  bool enable_xgrammar = false;

  // Max total kv tokens per chunk when chunked KV is enabled. Must be >= max_seq_len
  // so that any single request's tokens can always fit in one chunk
  size_t max_chunked_kv_tokens = 0;

  // ADP Balance Strategy configuration
  AttentionDPGroupBalanceConfig attention_dp_lb_config;

  // Ratio of SWA (Sliding Window Attention) blocks relative to the total device block count.
  // swa_block_num = device_block_num * swa_block_num_ratio.
  // 0.0 means SWA block allocator is disabled.
  float swa_block_num_ratio = 0.0f;
};

struct AllocatorConfig {
  // The preallocated blocks.
  size_t blocks_num = 0;

  // The block size, in bytes.
  size_t block_size = 0;

  // The max token number of one block.
  size_t block_token_num;

  MemoryDevice device;
};

struct BlockManagerConfig {
  // The config of allocator for cpu/gpu/npu.
  AllocatorConfig host_allocator_config;
  AllocatorConfig device_allocator_config;

  // The config of SWA (Sliding Window Attention) device block allocator.
  // block_size is set from the "SWA-cache" layout; blocks_num is derived from
  // device_block_num * swa_block_num_ratio in CalculateBlockNumber.
  AllocatorConfig swa_device_allocator_config;

  // The ratio of reserved device memory.
  float reserved_device_memory_ratio = 0.05;

  // The ratio of block device memory. use all left memory if less than 0.0.
  float block_device_memory_ratio = -1.0;

  // The scale fator of block host memory.
  float block_host_memory_factor = 10.0;

  // The ratio of dynamic reusable memory.
  float dynamic_reusable_memory_ratio = 1.0;

  // The size of host memory pool in GB, 0 means disabled.
  int host_pool_size_gb = 0;
};

// For cached manager, used for auto-prefix-caching.
struct CacheManagerConfig {
  // The token number of every block, not changed after created.
  size_t block_token_num = 16;

  // The tp num, cache manager use this to allocat blocks for every token.
  size_t tensor_para_size = 2;

  // The minimum consecutive length of flexible cache instances that can be
  // queried.
  size_t min_flexible_cache_num = 0;

  // Whether enable prefix caching.
  bool enable_prefix_caching = false;

  // Sliding window size for SWA (Sliding Window Attention) KV cache.
  // 0 means SWA is disabled (no sliding window, keep all KV blocks).
  size_t sliding_window_size = 0;

  // When true, SWACacheManager operates in non-contiguous mode: only the trailing
  // minimum_swa_block_num FA blocks in each allocation batch are bound to SWA cache
  // blocks.  When false (default), every new FA block gets an SWA cache block.
  bool enable_non_contiguous_swa_mode = false;
};

// For multiple node pipeline.
struct PipelineConfig {
  std::string master_host;
  uint16_t master_port;

  // Default for standalone mode.
  size_t world_size = 1;
  size_t node_rank = 0;

  // layer id range.
  int16_t lower_layer_idx = -1;
  int16_t upper_layer_idx = -1;

  // netxn layer id range.
  int16_t lower_nextn_layer_idx = -1;
  int16_t upper_nextn_layer_idx = -1;

  // The cache block num.
  // All pipeline nodes must be same.
  size_t device_block_num;
  size_t host_block_num;

  // The current port for data transfer.
  std::string data_host;
  uint16_t data_port;

  // The downstream data port for data transfer.
  std::string downstream_host;
  uint16_t downstream_port;

  // The nccl unique_id.
  char nccl_unique_id[128];

  DistributedCommunicationType pipeline_para_comm_type = DistributedCommunicationType::DEFAULT;

  void SetDistributeRelatedConfig() {
    const char *master_host_env = std::getenv("MASTER_HOST");
    const char *master_port_env = std::getenv("MASTER_PORT");
    const char *world_size_env = std::getenv("WORLD_SIZE");
    const char *node_rank_env = std::getenv("NODE_RANK");

    world_size = world_size_env ? std::stoi(world_size_env) : 1;
    node_rank = node_rank_env ? std::stoi(node_rank_env) : 0;
    if (world_size > 1) {
      if (!master_host_env || !master_port_env) {
        throw std::runtime_error(
            "The environment variable MASTER_HOST and MASTER_PORT must be set in distributed mode.");
      }
    }

    master_host = master_host_env ? master_host_env : "";
    master_port = master_port_env ? std::stoi(master_port_env) : 0;

    KLLM_LOG_INFO << "Initialize pipeline config, master_host:" << master_host << ", master_port:" << master_port
                  << ", world_size:" << world_size << ", node_rank:" << node_rank;
  }
};

struct ExpertParallelConfig {
  // Maser node info for expert parallelism.
  std::string expert_master_host;
  uint16_t expert_master_port;

  // Default for standalone mode.
  size_t expert_world_size = 1;
  size_t expert_para_size = 1;
  size_t expert_node_rank = 0;
  // expert_tensor_para_size = tensor_para_size / expert_para_size;
  size_t expert_tensor_para_size = 1;
  // expert_global_para_size = expert_para_size * expert_world_size;
  size_t global_expert_para_size = 1;

  size_t local_num_experts = 1;

  // I.E. expert_para_size = 4, the local_expert_rank = {0, 1, 2, 3}
  // tensor_para_size = 8, expert_para_size = 4, device_id = 0,1,2,3...,7
  // local_expert_rank = device_id % expert_para_size
  // When to init?
  size_t local_expert_rank = 0;

  // Node info of every node.
  std::string data_host;
  uint16_t data_port;

  // The downstream data port for data transfer.
  std::string downstream_host;
  uint16_t downstream_port;

  // The data port for data transfer of other expert nodes.
  std::vector<std::string> expert_node_host;
  std::vector<uint16_t> expert_node_port;

  // Store <expert_id, ep_node_rank>.
  std::map<uint32_t, uint32_t> expert_route_table;
  std::vector<uint32_t> local_expert_rank_route;
  // Expert_id on the current node.
  std::vector<uint32_t> local_experts;

  bool enable_expert_para;
  bool use_tcp = false;

  // The nccl unique_id.  [node_rank][nccl_id]
#ifdef ENABLE_CUDA
  std::vector<std::array<char, sizeof(ncclUniqueId)> > nccl_unique_ids;
  char nccl_unique_id[sizeof(ncclUniqueId)];
#endif

  // Fix later @xingjinglu
  DistributedCommunicationType expert_para_comm_type = DistributedCommunicationType::DEFAULT;
};

// The config of dsv4 compressor.
struct CompressorConfig {
  DataType state_dtype = TYPE_BF16;  // state storage type
};

// The config of attention backend.
struct AttnBackendConfig {
  bool enable_blocked_multi_token_forwarding_kv = false;
  bool use_flashinfer_for_decode = false;
  DataType kv_cache_dtype;    // kv_cache storage type
  size_t block_token_num{0};  // The max token number of one block.
  size_t block_size{0};       // The block size, in bytes.

  // "kv_cache.dtype" specified in the yaml config
  std::string kv_cache_dtype_str = "auto";

  // User preference for FlashAttention implementation selection.
  enum class FlashAttnImplChoice {
    AUTO = 0,  // Auto-detect by hardware and availability (default)
    FA3,       // FlashAttention 3
    VLLM_V26,  // vLLM FlashAttention 2.6+
    FA2_V26,   // FlashAttention 2.6+
    FA2_V25    // FlashAttention 2.5+
  };
  FlashAttnImplChoice flash_attn_impl_choice = FlashAttnImplChoice::AUTO;
};

struct ParallelismBasicConfig {
  size_t tensor_parallel_size{1};
  size_t attn_data_parallel_size{1};
  size_t attn_tensor_parallel_size{1};  // Determined by tp/dp
  size_t expert_parallel_size{1};
  size_t expert_world_size{1};
  size_t moe_tensor_para_size{1};
};

enum W4AFP8_MOE_BACKEND { Default = 0, GroupTriton = 1, TensorTriton = 2 };

enum FP4_MOE_BACKEND_TYPE { FP4Triton = 0, FP4Marlin = 1, FP4HummingW4A16 = 2, FP4HummingW4A8 = 3 };

// The communication mode for MoE all-to-all dispatch/combine.
enum class MoeAllToAllMode {
  DISABLED = 0,           // All-to-all disabled (default, use AllReduce-based communication)
  DEEPEP_NORMAL = 1,      // Use DeepEP normal-latency communication
  DEEPEP_LOW_LATENCY = 2  // Use DeepEP low-latency communication
};

// Config info used during runtime
// Some configs are determined by ModelConfig and BatchSchedulerConfig
struct RuntimeConfig {
  // Group 1: parallelism config
  ParallelismBasicConfig parallel_basic_config;

  // Group 2: execution graph config
  // For attention backend.
  AttnBackendConfig attn_backend_config;
  bool enable_full_shared_expert = false;
  bool separate_prefill_decode = false;
  CompressorConfig compressor_config;

  // True when this node is in PD-disagg DECODE role (no prefill computation on this node).
  // 用于 forwarding buffer 缩减：decode-only 节点的单 step token 数远小于 max_step_token_num，
  // 把多余的 buffer 让给 KV cache 块以提高并发。
  bool is_decode_only = false;

  bool enable_prefix_caching = false;  // Whether enable prefix caching.
  bool enable_flexible_caching = false;

  // EPLB related configurations
  bool enable_dump_eplb_data = false;  // Whether to dump EPLB topk_ids data
  std::string dump_eplb_path;          // Path to dump EPLB data, default: ~/.cache/KsanaLLM/EPLB/

  bool enable_load_eplb_config = false;  // Whether to load EPLB config
  std::string eplb_config_path;          // Path to EPLB json config file

  // Backend type of w4afp8 moe
  W4AFP8_MOE_BACKEND w4afp8_moe_backend = W4AFP8_MOE_BACKEND::Default;

  // Backend type of FP4 MoE (for DeepSeek V4 FP4 models)
  FP4_MOE_BACKEND_TYPE fp4_moe_backend = FP4_MOE_BACKEND_TYPE::FP4HummingW4A8;

  // Whether to normalize q and k before rotary position embedding in attention.
  // bool enable_qk_pre_norm_before_rotary_pos = false;

  // Schedule related. determined by schedule configs and cache related configs.
  size_t max_pp_batch_num{1};  // max number of batchs in pipeline parallel.
  int max_batch_size;
  size_t max_seq_len;            // The max token number of a sequence
  size_t max_step_token_num;     // The max token number of step
  size_t max_chunked_kv_tokens;  // The max total kv tokens per chunk when chunked KV is enabled

  size_t mtp_step_num = 0;

  bool enable_speculative_decoding = false;
  std::string speculative_method;

  bool enable_async = false;

  // Executor 端 build-ahead + device 侧 token 替换总开关（配置意图）。
  // 运行时是否真正启用由 FastPathController 根据全部硬条件决定。
  bool enable_executor_build_ahead = false;

  // TODO(robertyuan): No body set it?
  bool embed_tokens_use_cpu{false};  // Embed_tokens gather operation is processed on the CPU.

  // data type of intermediate data: input data and output data type of kernels
  DataType inter_data_type;

  bool enable_o_proj_out_of_dp = false;  // Whether to enable out-of-data-parallelism for o_proj in attention.

  // MOE topk score threshold, tokens with score below this threshold will be filtered
  float moe_topk_score_threshold = 0.0f;

  // Communication mode for MoE all-to-all dispatch/combine
  MoeAllToAllMode moe_all_to_all_mode = MoeAllToAllMode::DISABLED;

  // Maximum token number used for DeepEP low-latency buffer sizing.
  // When > 0, this replaces max_step_token_num as the buffer coefficient in LowLatency mode.
  // Configured via setting.global.deepep_low_latency_max_tokens in YAML config file.
  // Default: 256. When set to 0, falls back to max_step_token_num.
  size_t deepep_low_latency_max_tokens = 256;

  bool is_profile_mode = false;  // Only used for profiling performance

  // Non-contiguous SWA cache mode controls for DSV4 flash attention.
  // enable_non_contiguous_swa_mode=false: force-disable buffer path (rollback).
  // force_non_contiguous_swa_mode=true:  force-enable buffer path for all flash batches (debug).
  bool enable_non_contiguous_swa_mode = true;
  bool force_non_contiguous_swa_mode = false;
};

class ScheduleConfigParser {
 public:
  ScheduleConfigParser();

  // Parse environment from YAML reader.
  Status ParseScheduleConfig(YamlReader &yaml_reader, const ModelConfig &model_config);

  void Reset();

  Status UpdateModelConfig(ModelConfig &model_config);

  void UpdateMembers(const std::string &model_dir, ModelConfig &model_config);

  // Get the config of batch manager.
  Status GetBatchSchedulerConfig(BatchSchedulerConfig &batch_scheduler_config);
  void SetBatchSchedulerConfig(BatchSchedulerConfig &batch_scheduler_config);

  // Get the config of cached manager.
  Status GetCacheManagerConfig(CacheManagerConfig &cache_manager_config);

  void SetCacheManagerConfig(CacheManagerConfig &cache_manager_config);

  Status GetRuntimeConfig(RuntimeConfig &runtime_config);

  // Whether the auto-prefix-caching is enabled.
  bool IsPrefixCachingEnabled();

  size_t GetTransferLayerChunkSize();

  // Get the config of block manager.
  Status GetBlockManagerConfig(BlockManagerConfig &block_manager_config);

  // TODO(yancyliu): remove from here later.
  void SetBlockManagerConfig(const BlockManagerConfig &block_manager_config);
  Status CalculateBlockNumber(size_t device_free, size_t device_total);
  Status ResetPipelineBlockNumber();
  size_t GetTotalDeviceBlockNum();
  size_t GetTotalHostBlockNum();
  std::vector<int> GetDataParaGroupDevices(int dp_id);

  void SetTensorParallelSize(size_t tensor_parallel_size) {
    runtime_config_.parallel_basic_config.tensor_parallel_size = tensor_parallel_size;
  }

  void SetAttnDataParallelSize(size_t attn_data_parallel_size) {
    runtime_config_.parallel_basic_config.attn_data_parallel_size = attn_data_parallel_size;
  }

  void SetExpertParallelSize(size_t expert_parallel_size) {
    runtime_config_.parallel_basic_config.expert_parallel_size = expert_parallel_size;
  }

  size_t GetMaxBatchSize() const { return batch_scheduler_config_.max_batch_size; }

  // Modify reserved_device_memory_ratio
  void SetReservedDeviceRatio(float reserved_device_memory_ratio);

  // Set and get multiple node pipeline config.
  void SetPipelineConfig(const PipelineConfig &pipeline_config) { pipeline_config_ = pipeline_config; }

  Status GetPipelineConfig(PipelineConfig &pipeline_config) const {
    pipeline_config = pipeline_config_;
    return Status();
  }

  void GetAttnBackendConfig(AttnBackendConfig &attn_backend_config) {
    attn_backend_config = runtime_config_.attn_backend_config;
  }

  void SetAttnBackendConfig(const AttnBackendConfig &attn_backend_config) {
    runtime_config_.attn_backend_config = attn_backend_config;
  }

  Status GetExpertParallelConfig(ExpertParallelConfig &expert_parallel_config) const {
    expert_parallel_config = expert_parallel_config_;
    return Status();
  }

  void SetExpertParallelConfig(const ExpertParallelConfig &expert_parallel_config) {
    expert_parallel_config_ = expert_parallel_config;
  }

  Status GetConnectorConfigs(ConnectorConfig &connector_config) const {
    // 检查connector_config_是否已初始化 - group_role
    // pd_v2 (is_pd_v2 = true) 不依赖 simple_router; 走 Mooncake metadata
    // 自发现, 因此 router_addr 可以为空. 只要 group_role 设了就视为初始化.
    if (connector_config_.group_role == GroupRole::NONE) {
      return Status(RET_CONFIG_NOT_FOUND, "Connector config is not initialized.");
    }
    if (!pd_v2_runtime_config_.is_pd_v2 && connector_config_.router_addr.empty()) {
      return Status(RET_CONFIG_NOT_FOUND, "Connector config is not initialized (V1 needs router_addr).");
    }
    connector_config = connector_config_;
    return Status();
  }

  void SetConnectorConfigs(const ConnectorConfig &connector_config) {
    connector_config_ = connector_config;
    return;
  }

  // Init disaggregating prefill and decode connector config
  void InitConnectorConfig(YamlReader &yaml_reader);

  // pd_v2 runtime config accessors. Populated by InitConnectorConfig as a
  // twin write alongside connector_config_ when backend == "v2". Callers
  // on the pd_v2 path read this; callers on the V1 path keep reading
  // connector_config_. See pd_v2_runtime_config.h for rationale.
  Status GetPdV2RuntimeConfig(PdV2RuntimeConfig &config) const {
    if (!pd_v2_runtime_config_.is_pd_v2) {
      return Status(RET_CONFIG_NOT_FOUND, "pd_v2 runtime config is not initialized.");
    }
    config = pd_v2_runtime_config_;
    return Status();
  }

  void SetPdV2RuntimeConfig(const PdV2RuntimeConfig &config) { pd_v2_runtime_config_ = config; }

  Status GetGlobalCacheConnectorConfig(GlobalCacheConnectorConfig &config) const {
    if (global_cache_connector_config_.service_endpoint.empty()) {
      return Status(RET_CONFIG_NOT_FOUND, "Global cache connector config is not initialized.");
    }
    config = global_cache_connector_config_;
    return Status();
  }

  void SetGlobalCacheConnectorConfig(const GlobalCacheConnectorConfig &config) {
    global_cache_connector_config_ = config;
  }

  void InitGlobalCacheConnectorConfig(YamlReader &yaml_reader);

  // Calculate block size via model configs.
  Status InitializeBlockManagerConfig(const ModelConfig &model_config,
                                      std::shared_ptr<BaseModelCacheLayout> block_cache_layout);

  // Init Expert-Parallel Config from env.
  void InitializeExpertParallelConfig();

  // Init KV cache configs for all attention modules
  void InitializeKVCacheConfigs(const ModelConfig &model_config, const PipelineConfig &pipeline_config,
                                std::shared_ptr<BaseModelCacheLayout> block_cache_layout);

 private:
  // The config of batch schedule.
  BatchSchedulerConfig batch_scheduler_config_;

  // The config used by cache manager.
  CacheManagerConfig cache_manager_config_;

  // The config of block manager.
  BlockManagerConfig block_manager_config_;

  RuntimeConfig runtime_config_;

  // TODO(robertyuan): This two configs will be set by data channel, fix them later
  // For distributed multiple node pipeline.
  PipelineConfig pipeline_config_;
  // For expert parallel.
  ExpertParallelConfig expert_parallel_config_;

  // Store parsed connector configurations
  ConnectorConfig connector_config_;

  // pd_v2 runtime config — populated alongside connector_config_ when
  // setting.connector.backend == "v2". pd_v2 read-sites consult this
  // instead of connector_config_ to avoid V1 derivations leaking
  // (e.g. K7 — see pd_v2_runtime_config.h).
  PdV2RuntimeConfig pd_v2_runtime_config_;

  // Store global cache connector configuration
  GlobalCacheConnectorConfig global_cache_connector_config_;
};

}  // namespace ksana_llm
