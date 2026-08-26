/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "models/base/attention_meta/attention_meta_creator.h"
#include "models/common/common_model.h"
#include "models/kimi_k3/attention_meta/kimi_k3_attention_meta_creator.h"
#include "modules/attention/multihead_latent_attention.h"
#include "modules/basic/add.h"
#include "modules/basic/attn_res.h"
#include "modules/basic/layernorm.h"
#include "modules/basic/linear.h"
#include "modules/basic/moe.h"
#include "modules/ffn/shared_expert_layer.h"
#include "modules/ffn/two_layered_ffn.h"

namespace ksana_llm {

class KimiK3DecoderLayer {
 public:
  KimiK3DecoderLayer(int layer_idx, LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config,
                     MlaBuffers& mla_buffers, TensorBuffer* moe_buffer);

  ~KimiK3DecoderLayer() = default;

  // prefix_buffer: running residual sum, shape [num_tokens, hidden]
  // residual_buffer: residual snapshot bank, shape [num_tokens, num_blocks, hidden]
  Status Forward(std::vector<Tensor>& prefix_buffer, std::vector<Tensor>& residual_buffer,
                 const bool is_multi_token_forward, ForwardingContext& forwarding_context, bool is_first_layer);

 private:
  // Dispatches to MLA or KDA depending on this layer's type.
  Status CommonAttn(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                    const bool is_multi_token_forward, ForwardingContext& forwarding_context);

  // Dispatches to MLP or MoE depending on this layer's type.
  Status CommonMlp(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                   const bool is_multi_token_forward, ForwardingContext& forwarding_context);

 private:
  bool is_block_write_layer_;
  bool enable_full_shared_expert_;
  int layer_idx_;
  int rank_;

  std::unique_ptr<AttnRes> attn_res_;
  std::unique_ptr<AttnRes> mlp_res_;

  std::unique_ptr<Add> add_;
  std::shared_ptr<TpCommunicator> tp_comm_;

  // Either mla_ or kda_ is non-null, per kimi_linear_config.kda_layers.
  std::unique_ptr<MultiHeadLatentAttention> mla_;

  // Either mlp_ (dense layer) or moe_ + shared_mlp_ + expert_gate_ (MoE layer) is non-null,
  // per moe_config.first_k_dense_replace.
  std::unique_ptr<TwoLayeredFFN> mlp_;
  std::shared_ptr<SharedExpertLayer> shared_expert_;
  std::shared_ptr<Linear> expert_gate_;
  std::shared_ptr<MoE> moe_;

  std::shared_ptr<Linear> routed_expert_down_proj_;
  std::shared_ptr<Layernorm> routed_expert_norm_;
  std::shared_ptr<Linear> routed_expert_up_proj_;

  MlaBuffers& mla_buffers_;
  TensorBuffer* const moe_buffer_;
};

class KimiK3Model : public CommonModel {
 public:
  KimiK3Model(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
              std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
              std::shared_ptr<BaseModelCacheLayout> block_cache_layout);

  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;

  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

  // Register slot resource (mamba slot pool).
  void RegisterInflightResources(InflightResourceManager& mgr) override;

 protected:
  using CommonModel::GetHiddenUnitBufferTensors;

  std::unique_ptr<AttentionMetaCreator> CreateCustomAttentionMetaCreator(
      const AttentionMetaCreator::Config& config) override {
    return std::make_unique<KimiK3AttentionMetaCreator>(config);
  }

  std::unordered_map<int, std::shared_ptr<KimiK3DecoderLayer>> layers_;
  std::unique_ptr<AttnRes> output_attn_res_;

  MlaBuffers mla_buffers_;
  TensorBuffer* moe_buffer_;
};

}  // namespace ksana_llm
