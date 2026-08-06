/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/dspark_layer.h"
#include "models/base/attention_meta/attention_meta_creator.h"
#include "models/common/common_model.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta_creator.h"
#include "modules/attention/dsv4_mqa.h"
#include "modules/basic/add.h"
#include "modules/basic/dspark_markov_head.h"
#include "modules/basic/layernorm.h"
#include "modules/basic/linear.h"
#include "modules/basic/mhc.h"
#include "modules/basic/moe.h"
#include "modules/ffn/shared_expert_layer.h"
#include "modules/ffn/two_layered_ffn.h"

namespace ksana_llm {
class DeepSeekV4DecoderLayer {
 public:
  DeepSeekV4DecoderLayer(int layer_idx, bool is_moe, LayerCreationContext& creation_context,
                         ModelCreationConfig& model_creation_config, MhcBuffers& mhc_buffers,
                         Dsv4AttnBuffer& dsv4_attn_buffer, TensorBuffer* moe_buffer,
                         TensorBuffer* shared_expert_inter_buffer = nullptr,
                         TensorBuffer* shared_expert_output_buffer = nullptr);

  ~DeepSeekV4DecoderLayer() = default;

  Status Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                 ForwardingContext& forwarding_context);

 private:
  Status CommonMlp(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                   const bool is_multi_token_forward, ForwardingContext& forwarding_context);

 private:
  bool is_moe_;

  std::unique_ptr<Mhc> pre_attention_mhc_;
  std::unique_ptr<Mhc> post_attention_mhc_;
  std::shared_ptr<Add> add_;
  std::shared_ptr<TpCommunicator> tp_comm_;

  std::shared_ptr<Dsv4Mqa> mqa_;

  const bool enable_full_shared_expert_;
  int layer_idx_;
  int rank_;
  uint32_t num_hash_layers_ = 0;

  std::shared_ptr<TwoLayeredFFN> mlp_;
  std::shared_ptr<SharedExpertLayer> shared_expert_;
  std::shared_ptr<Linear> expert_gate_;
  std::shared_ptr<MoE> moe_;

  MhcBuffers& mhc_buffers_;
  TensorBuffer* const moe_buffer_;
};

class DeepSeekV4MtpLayer {
 public:
  DeepSeekV4MtpLayer(const int layer_idx, LayerCreationContext& creation_context,
                     ModelCreationConfig& model_creation_config, std::shared_ptr<DeepSeekV4DecoderLayer> decoder_layer);

  ~DeepSeekV4MtpLayer() = default;

  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context);

 private:
  const size_t hc_mult;
  const size_t hidden_dim;

  std::unique_ptr<Layernorm> enorm_;
  std::unique_ptr<Layernorm> hnorm_;
  std::unique_ptr<Linear> e_proj_;
  std::unique_ptr<Linear> h_proj_;
  std::unique_ptr<BaseLayer> gather_layer_;
  std::unique_ptr<Add> add_;

  std::shared_ptr<DeepSeekV4DecoderLayer> decoder_layer_;
};

// DSpark speculative layer forward declaration.
class DeepSeekV4DSparkLayer;

class DeepSeekV4Model : public CommonModel {
 public:
  DeepSeekV4Model(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                  std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
                  std::shared_ptr<BaseModelCacheLayout> block_cache_layout);

  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;

  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

  Status ForwardHead(std::vector<ForwardRequest>& forward_reqs, size_t step_index) override;

  // DSpark step: return current step's [B, V] row base when set by ForwardHead;
  // otherwise fall back to the model's whole logits base.
  float* GetLogitsPtr() override {
    return dspark_step_logits_ptr_ ? dspark_step_logits_ptr_ : CommonModel::GetLogitsPtr();
  }

  // Register compressor slot resource (and any future DSv4 inflight resources).
  void RegisterInflightResources(InflightResourceManager& mgr) override;

 protected:
  using CommonModel::GetHiddenUnitBufferTensors;
  using CommonModel::mhc_buffers_;

  std::unique_ptr<AttentionMetaCreator> CreateCustomAttentionMetaCreator(
      const AttentionMetaCreator::Config& config) override {
    return std::make_unique<DeepSeekV4AttentionMetaCreator>(config);
  }

  std::unordered_map<int, std::shared_ptr<DeepSeekV4DecoderLayer>> layers_;
  std::unordered_map<int, std::shared_ptr<DeepSeekV4MtpLayer>> nextn_layers_;
  std::unordered_map<int, std::shared_ptr<DeepSeekV4DSparkLayer>> dspark_layers_;

  // DSpark markov bias generator.
  std::unique_ptr<DSparkMarkovHead> dspark_markov_head_;

  // DSpark kernel wrapper.
  std::unique_ptr<DSparkLayer> dspark_layer_;

  // DSpark single-step markov input.
  Tensor dspark_step_token_ids_;
  std::vector<int> dspark_step_token_ids_host_;
  // DSpark cross-layer hidden state capture buffer.
  Tensor dspark_hidden_tensor_;
  // next-step hidden buffer for MTP (owned by model, injected into ForwardingBuffers).
  std::vector<Tensor> next_step_hidden_tensors_;
  size_t dspark_block_size_ = 0;

  // Base address of the current step's [batch_size, V] logits row within logits_tensor.
  float* dspark_step_logits_ptr_ = nullptr;

 public:
  static constexpr int kNextnDecoderLayerNum = 1;  // Each token generated by MTP use 1 decoder layer
  int16_t nextn_layer_idx_;                        // Record the index of MTP layers used in this forward pass.

  const int first_k_dense_replace_;

  // Shared pre-allocated attention buffers for DSV4 MQA (kv_compressed_output, topk_indices).
  Dsv4AttnBuffer dsv4_attn_buffer_;

  TensorBuffer* moe_buffer_;

  // Independent buffers for shared expert multi-stream parallel execution.
  // Shared across all layers (layers execute sequentially).
  TensorBuffer* shared_expert_inter_buffer_ = nullptr;
  TensorBuffer* shared_expert_output_buffer_ = nullptr;
};
}  // namespace ksana_llm
