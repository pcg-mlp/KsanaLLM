/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <mutex>
#include <unordered_map>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/context.h"
#include "configure/environment.h"
#include "layers/add_layer.h"
#include "layers/assemble_tokens_hidden_layer.h"
#include "layers/base_layer.h"
#include "layers/cast_layer.h"
#include "layers/emb_lookup_layer.h"
#include "layers/flash_attention_layer.h"
#include "layers/input_refit_layer.h"
#include "layers/layernorm_layer.h"
#include "layers/matmul_layer_factory.h"
#include "layers/paged_attention_layer.h"
#include "layers/silu_mul_layer.h"
#include "models/base/base_model.h"
#include "models/base/forwarding_context.h"
#include "models/base/model_communicator.h"
#include "models/base/model_input.h"
#include "models/base/model_output.h"
#include "models/common_moe/moe_config.h"
#include "models/communicator/tp_communicator.h"
#include "models/llama/llama_weight.h"
#include "runtime/hidden_unit_buffer.h"
#include "runtime/infer_stage.h"
#include "utils/optional_file.h"
#include "utils/status.h"
#include "utils/tensor.h"
#include "utils/utils.h"

#include "modules/basic/layernorm.h"
#include "modules/basic/linear.h"
#include "modules/basic/mhc.h"

namespace ksana_llm {

// The layernorm position type.
enum class LayerNormPosition { PRE_NORM = 0, POST_NORM = 1 };

// Describe the model architecture.
struct ModelRunConfig {
  // The model position embedding.
  PositionEncoding position_encoding = PositionEncoding::ROPE;

  // Use pre-norm or post-norm.
  LayerNormPosition layernorm_position = LayerNormPosition::PRE_NORM;

  // If use rotary_embedding_pos for embedding lookup
  bool emb_lookup_use_rotary_embedding_pos = false;

  // Whether word embedding uses emb_scale.
  bool use_emb_scale = false;
  // The word embedding scale factor.
  float emb_scale = 1.f;
  // Scaling the hidden states of residual connections.
  float scale_depth = 1.f;
  // Whether to return hidden states before the final layer normalization.
  bool return_hidden_states_before_norm = true;
};

// A common implement
class CommonModel : public BaseModel {
 public:
  CommonModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
              std::shared_ptr<Context> context);
  ~CommonModel() override;

  // Initialize the run config.
  void InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight,
                     std::shared_ptr<BaseModelCacheLayout> block_cache_layout);

  float* GetLogitsPtr() override;

  // refer
  // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L942
  Status Forward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs,
                 HiddenUnitDeviceBuffer* hidden_unit, RunMode run_mode = RunMode::kMain, size_t model_input_slot = 0,
                 const int64_t* prev_ring_dev = nullptr, bool gpu_input_already_prepared = false) override;

  Status PrepareModelInputGpu(std::vector<ForwardRequest>& forward_reqs, RunMode run_mode,
                              size_t model_input_slot) override;

  // 取本步 slot ModelInput 的 cur_batch_pairs_tensor 设备指针 + batch_size，供 Sampler::WriteRing 使用。
  void GetCurBatchPairs(size_t model_input_slot, const int64_t** out_pairs_dev, int* out_batch_size) override;

  // Propagate inflight resource manager reference down to ForwardingContext.
  void SetInflightResourceManager(InflightResourceManager* mgr) override;

  // 设置/清除 ForwardingContext 中的 ProfileMetrics 及当前子 metrics 指针。
  void SetForwardProfileMetrics(ProfileMetrics* pm, ForwardPerfMetrics* current) override;

  // Update response. Stop inference when the return value is true.
  bool UpdateResponse(std::vector<ForwardRequest>& forward_reqs, Tensor& output, const std::string& stage);

 private:
  virtual Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) = 0;

 private:
  // Execute the embedding lookup.
  Status LookupEmbedding(ForwardingContext& forwarding_context, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                         std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode = RunMode::kMain);

  // Execute the forward of specific layers.
  virtual Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) = 0;

  // Execute the lm head, and generate the logits.
  virtual Status LmHead(ForwardingContext& forwarding_context, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                        std::vector<ForwardRequest>& forward_reqs, RunMode run_mode);

 protected:
  // Get tensors hidden buffer.
  std::vector<Tensor>& GetHiddenUnitBufferTensors(ForwardingContext& forwarding_context);
  void SetHiddenUnitBufferTensors(HiddenUnitDeviceBuffer* const hidden_unit);

 public:
  using BaseModel::context_;
  using BaseModel::rank_;

  // Whether auto prefix caching is enabled.
  bool prefix_caching_enabled_;

  // Check if speculative decoding is enabled
  bool speculative_decoding_enabled_;

  // The model config.
  ModelConfig model_config_;

  RuntimeConfig runtime_config_;

  // The pipeline_config for distributed mode.
  PipelineConfig pipeline_config_;

  // The expert parallel config for multi nodes.
  ExpertParallelConfig expert_parallel_config_;
  // The model run config.
  ModelRunConfig model_run_config_;

  std::shared_ptr<BaseLayer> emb_lookup_layer_;
  std::shared_ptr<BaseLayer> cpu_emb_lookup_layer_;

  std::shared_ptr<BaseLayer> assemble_tokens_hidden_layer_;
  std::shared_ptr<BaseLayer> cast_layer_;
  std::shared_ptr<InputRefitLayer> input_refit_layer_;

  std::shared_ptr<Linear> lm_head_;
  std::shared_ptr<Layernorm> lm_head_prenorm_{nullptr};

  // TODO(robertyuan): layer_creation_context_ should be deleted after layer creation.
  // However, matmul_layer_factory will delete the buffer during destroying.
  // Fix this after CommonModel is deleted.
  LayerCreationContext layer_creation_context_;

  ModelBuffers model_buffers_;
  // Single forwarding context shared by all Forward() callers on this rank.
  // Mutated state inside (ModelInput, *_meta, ForwardingBuffers, ModelOutput) is NOT thread-safe,
  // so concurrent Forward() invocations must be serialized via forward_mutex_ below.
  std::unique_ptr<ForwardingContext> forwarding_context_ = nullptr;

  // Serializes CommonModel::Forward(). On the sampling rank both ModelRunner::HandleExecute (main
  // forward) and ModelRunner::HandlePostprocess -> ExecutorRuntime::MtpForward (MTP forward) can
  // drive Forward() against the same forwarding_context_; under max_pp_batch_num >= 2 those calls
  // overlap and race on the shared host vectors / TensorBuffer in-use flags. This mutex enforces
  // the single-writer invariant the rest of the model layer assumes. Non-sampling ranks acquire it
  // unconditionally but are uncontended.
  std::mutex forward_mutex_;

  // Be a replacement of residual_buffer_, for distributed mode only.
  std::vector<Tensor> distributed_device_buffer_;
  std::vector<Tensor> distributed_device_buffer_prefill_;

  Tensor cpu_input_tokens_tensor_;
  Tensor cpu_tokens_emb_tensor_;

  // Only used for QWenVL
  Tensor mrotary_section_tensor_;
  // Only used for arc_hunyuan_video
  Tensor xdrotary_section_tensor_;

 protected:
  bool IsPrefixCachingComputationReuse();

  // Override to return a model-specific AttentionMetaCreator.
  // Called once in InitRunConfig() right after ForwardingContext is initialized.
  // The default implementation returns nullptr (keep the standard creator built in ModelInput).
  virtual std::unique_ptr<AttentionMetaCreator> CreateCustomAttentionMetaCreator(
      const AttentionMetaCreator::Config& config) {
    return nullptr;
  }

  Status EmbedTokensUseCpu(Tensor& embedding_weight, std::vector<ForwardRequest>& forward_reqs,
                           ForwardingContext& forwarding_context);

  virtual Status EmbedTokensUseGpu(Tensor& embedding_weight, ForwardingContext& forwarding_context);

  // Only set by required models
  std::unique_ptr<Mhc> initial_mhc_;
  std::unique_ptr<Mhc> final_mhc_;
  MhcBuffers mhc_buffers_;

  // When set, LmHead uses these for kNextN instead of final_mhc_/lm_head_prenorm_.
  std::unique_ptr<Mhc> nextn_final_mhc_;
  std::unique_ptr<Layernorm> nextn_shared_head_norm_;
};

}  // namespace ksana_llm
