/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <vector>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "models/base/layer_creation_context.h"
#include "profiler/profile_event.h"
#include "utils/tensor.h"

#include "models/base/model_communicator.h"
#include "models/base/model_input.h"
#include "models/base/model_output.h"
#include "profiler/sched_event_tracer.h"

namespace ksana_llm {

// 计算 decode-only 节点的 forwarding buffer 所需 token 数。
// decode-only 节点不做 prefill，单 step 产出的 token 数上界为
//   max_batch_size * max_decode_tokens_per_req * attn_data_parallel_size
// 其中 max_decode_tokens_per_req = mtp_step_num + 2 * ptp_step_num + 1（与 schedule_config_parser 一致），
// 已涵盖 MTP / PTP / generated token。
// 该值通常远小于 max_step_token_num（按 prefill chunk 分配），省下的显存可用于 KV block。
size_t ComputeDecodeOnlyStepTokenNum(size_t max_batch_size, size_t max_decode_tokens_per_req,
                                     size_t attn_data_parallel_size);

// 从 RuntimeConfig + BatchSchedulerConfig 中选择 forwarding buffer 生效的 token_num。
// - is_decode_only == true: 使用 ComputeDecodeOnlyStepTokenNum 的缩减值（参考 max_decode_tokens_per_req）
// - 其他情况: 使用 max_step_token_num（保持原行为）
size_t SelectEffectiveStepTokenNum(const RuntimeConfig& runtime_config,
                                   const BatchSchedulerConfig& batch_scheduler_config);

// 构造 decode-only buffer 缩减的日志文本，方便 ForwardingBuffers::Init 打印 & 单测校验。
std::string FormatDecodeOnlyShrinkLog(const RuntimeConfig& runtime_config,
                                      const BatchSchedulerConfig& batch_scheduler_config,
                                      size_t effective_token_num);

struct ForwardingBuffers {
  //  NOTE(karlluo): The following 4 buffers(3 different kinds) are used as temporary buffers during the whole model
  //  inference:
  //  1. intermedia buffer: `hidden_buffer_0` and `hidden_buffer_1` serve as the input and output for each layer.
  //     We assume that the input of each layer is taken from `hidden_buffer_0`, the output is
  //     put into `hidden_buffer_1`, and then swapped with `hidden_buffer_0`. This convention
  //     makes each layer independent and pluggable.
  //  2. operators' extra buffer: `shared_buffer` is shared to store the output of the up layer for gated activation
  //     (`gated_buffer_`), as the fixed input buffer for custom reduce sum (`reduce_buffer_`),
  //     and as the extra workspace for paged attention (`paged_buffer_`).
  //  3. kv cache buffer: `kv_cache_buffer` stores the key-value pairs for all attention layers
  //     across different sequence positions. This enables efficient autoregressive generation
  //     by avoiding recomputation of previously calculated key-value pairs during inference.
  //     The buffer supports both paged attention and continuous memory layouts for optimal
  //     memory utilization and access patterns.
  TensorBuffer* hidden_buffer_0;
  TensorBuffer* hidden_buffer_1;
  TensorBuffer* shared_buffer;
  TensorBuffer* kv_cache_buffer;

  // This buffer is used among multiple forward calls
  std::vector<Tensor> mtp_hidden_buffer_tensors;

  void Init(std::shared_ptr<Context> context, const int rank, const ModelConfig& model_config,
            const RuntimeConfig& runtime_config, BufferManager* const buffer_mgr);

  void CalculateBuffersShape(std::shared_ptr<Context> context, const size_t batch_size, const size_t max_token_num,
                             const DataType& weight_type);

  ModelConfig model_config;
  RuntimeConfig runtime_config;

  // Map to record each buffers shape.
  std::unordered_map<std::string, std::vector<size_t>> buffers_shape_map;
};

struct ModelBuffers {
  std::unique_ptr<ForwardingBuffers> buffers_;

  Tensor cos_sin_cache_tensor_;
  Tensor compress_cos_sin_cache_tensor_;

  void Init(std::shared_ptr<Context> context, int rank, const ModelConfig& model_config,
            const RuntimeConfig& runtime_config, BufferManager* buffer_mgr);
};

class ForwardingContext {
 public:
  ~ForwardingContext() {}
  void Init(std::shared_ptr<Context> context, int rank, const ModelConfig& model_config,
            const RuntimeConfig& runtime_config, const PipelineConfig& pipeline_config, ForwardingBuffers* buffers,
            BufferManager* buffer_mgr, std::shared_ptr<BaseModelCacheLayout> block_cache_layout);

  void UpdateBeforeForward(std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode);

  void UpdateAfterForward(std::vector<ForwardRequest>& forward_reqs);

  // Forwarder to ModelInput::EvictFinishedRequests; releases compressor slots for finished or
  // recomputed requests. Safe for ranks that do not use CompressorMeta.
  void EvictFinishedRequests(const std::vector<int64_t>& finished_req_ids);

 public:
  ForwardingBuffers* GetForwardingBuffers() { return buffers_; }

  AttentionForwardContext& GetAttentionForwardContext() { return attn_ctx_; }

  inline const size_t GetAttentionDataParallelSize() { return attn_data_parallel_size_; }

  std::shared_ptr<ModelCommunicator>& GetModelCommunicator() { return model_communicator_; }

  std::shared_ptr<ModelOutput>& GetModelOutput() { return model_output_; }

  std::shared_ptr<ModelInput>& GetModelInput() { return model_input_; }

  std::shared_ptr<BaseModelCacheLayout>& GetBlockCacheLayout() { return block_cache_layout_; }

  inline BatchRequestSchedInfo& GetBatchRequestSchedInfo() { return batch_event_info_; }

  inline const bool IsForwardingLayers() { return is_forwarding_layers_; }

  inline void SetIsForwardingLayers(const bool is_forwarding_layers) { is_forwarding_layers_ = is_forwarding_layers; }

  inline const std::shared_ptr<Context>& GetContext() { return context_; }

  inline void SetContext(std::shared_ptr<Context> context) { context_ = context; }

  inline int GetCurrentRank() const { return rank_; }

  inline void SetCurrentRank(const int rank) { rank_ = rank; }

  inline const PipelineConfig& GetPipelineConfig() { return pipeline_config_; }

 private:
  // Rank of current inference device
  int rank_;

  // Current inference context
  std::shared_ptr<Context> context_;

  // The model input information.
  std::shared_ptr<ModelInput> model_input_;

  // The model output.
  std::shared_ptr<ModelOutput> model_output_;

  // Used for tracing sched events.
  BatchRequestSchedInfo batch_event_info_;

  // The model communicator.
  std::shared_ptr<ModelCommunicator> model_communicator_;

  // Pipeline parallel configuration
  PipelineConfig pipeline_config_;

  // Attention data parallel size
  size_t attn_data_parallel_size_ = 1;

  // Extra attributes for attention
  AttentionForwardContext attn_ctx_;

  // Buffers for temporary buffers during the whole model inference
  ForwardingBuffers* buffers_;

  // The cache layout of every kv cache block.
  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_ = nullptr;

  // The original vocab size of the model
  size_t vocab_size_;

  // Vocab size aligned and padded with tensor_para_size
  size_t vocab_size_pad_;

  // mark state for sched event recording
  bool is_forwarding_layers_ = false;
};
}  // namespace ksana_llm
