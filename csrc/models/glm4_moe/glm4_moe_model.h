/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/common/model_interface.h"

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "models/communicator/tp_communicator.h"
#include "models/glm4_moe/glm4_moe_mtp_draft_model.h"
#include "modules/attention/multihead_attention.h"
#include "modules/basic/linear.h"
#include "modules/basic/moe.h"
#include "modules/ffn/two_layered_ffn.h"

namespace ksana_llm {

class Glm4MoeDecoderLayer {
 public:
  Glm4MoeDecoderLayer(int layer_idx, bool is_moe, TensorBuffer* moe_buffer, LayerCreationContext& creation_context,
                      ModelCreationConfig& model_creation_config);
  ~Glm4MoeDecoderLayer() = default;

  Status Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                 ForwardingContext& forwarding_context);

 private:
  Status ForwardMlp(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                    const bool is_multi_token_forward, ForwardingContext& forwarding_context);

 private:
  int layer_idx_;
  bool is_moe_;

  std::shared_ptr<Add> add_;
  std::shared_ptr<Layernorm> input_layernorm_;
  std::shared_ptr<Layernorm> post_attention_layernorm_;
  std::shared_ptr<TpCommunicator> tp_comm_;

  // Attention with QK Norm
  // Note: QK Norm is handled internally by MultiHeadAttention/CommonAttention
  std::shared_ptr<MultiHeadAttention> mha_;

  // MoE components (for MoE layers)
  std::shared_ptr<MoE> moe_;
  std::shared_ptr<Linear> expert_gate_;
  std::shared_ptr<TwoLayeredFFN> shared_mlp_;

  // FFN components (for dense layers)
  std::shared_ptr<TwoLayeredFFN> mlp_;

  // Non-owning pointer to MoE workspace buffer. Lifetime is managed by the parent Glm4MoeModel.
  TensorBuffer* moe_buffer_;
};

class Glm4MoeMtpLayer {
 public:
  Glm4MoeMtpLayer(int layer_idx, TensorBuffer* moe_buffer, LayerCreationContext& creation_context,
                  ModelCreationConfig& model_creation_config, std::shared_ptr<Glm4MoeDecoderLayer> decoder_layer);
  ~Glm4MoeMtpLayer() = default;

  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context);

 private:
  std::shared_ptr<Layernorm> enorm_;
  std::shared_ptr<Layernorm> hnorm_;
  std::shared_ptr<BaseLayer> concat_layer_;
  std::shared_ptr<Linear> eh_proj_;
  std::shared_ptr<BaseLayer> gather_layer_;
  std::shared_ptr<BaseLayer> emb_lookup_layer_;
  std::shared_ptr<Glm4MoeDecoderLayer> decoder_layer_;
  std::shared_ptr<TpCommunicator> tp_comm_;
};

class Glm4Moe : public ModelInterface {
 public:
  Glm4Moe() {}
  ~Glm4Moe() = default;

  Status GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) override;
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;
  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) override;

 private:
  // Non-owning pointer to MoE workspace buffer. Lifetime is managed by the parent Glm4MoeModel.
  TensorBuffer* moe_buffer_;
  int first_k_dense_replace_;  // First layer index that uses MoE

  std::map<int, std::shared_ptr<Glm4MoeDecoderLayer>> decoder_layers_;
};

class Glm4MoeModel : public CommonModel {
 public:
  Glm4MoeModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
               std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
               std::shared_ptr<BaseModelCacheLayout> block_cache_layout);
  ~Glm4MoeModel() = default;

 private:
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config);
  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel::GetHiddenUnitBufferTensors;

 private:
  Glm4Moe glm4moe_;

  // MTP (Multi-Token Prediction) support
  std::unordered_map<int, std::shared_ptr<Glm4MoeMtpLayer>> nextn_layers_;
  std::unique_ptr<Glm4MoeMtpDraftModel> draft_model_{nullptr};
  // Each MTP "step" uses exactly one decoder layer. If future models require multiple
  // decoder layers per MTP step, this constant should be updated accordingly.
  static constexpr size_t kNextnDecoderLayerNum = 1;
  int16_t nextn_layer_idx_ = -1;

  // next-step hidden buffer for MTP (owned by model, injected into ForwardingBuffers).
  std::vector<Tensor> next_step_hidden_tensors_;
};

}  // namespace ksana_llm
