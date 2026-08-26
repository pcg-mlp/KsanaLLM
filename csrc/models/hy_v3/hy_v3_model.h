/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/common/model_interface.h"

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "models/communicator/tp_communicator.h"
#include "modules/attention/multihead_attention.h"
#include "modules/basic/linear.h"
#include "modules/basic/moe.h"
#include "modules/ffn/two_layered_ffn.h"

namespace ksana_llm {

// One HunYuan V3 decoder layer.
// Attention: GQA + pre-RoPE QK-Norm, no QKV bias (same as Qwen3-MoE).
// FFN: dense TwoLayeredFFN for the first_k_dense prefix, otherwise a MoE block
// (sigmoid + expert-bias routing) plus a single shared expert (DeepSeek-V3 style).
class HyV3DecoderLayer {
 public:
  HyV3DecoderLayer(int layer_idx, bool is_moe, TensorBuffer* moe_buffer, LayerCreationContext& creation_context,
                   ModelCreationConfig& model_creation_config);
  ~HyV3DecoderLayer() = default;

  Status Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                 ForwardingContext& forwarding_context);

 private:
  Status ForwardMlp(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                    const bool is_multi_token_forward, ForwardingContext& forwarding_context);

  int layer_idx_;
  bool is_moe_;

  std::shared_ptr<Add> add_;
  std::shared_ptr<Layernorm> input_layernorm_;
  std::shared_ptr<Layernorm> post_attention_layernorm_;
  std::shared_ptr<TpCommunicator> tp_comm_;

  std::shared_ptr<MultiHeadAttention> mha_;

  // MoE components (MoE layers).
  std::shared_ptr<MoE> moe_;
  std::shared_ptr<Linear> expert_gate_;
  std::shared_ptr<TwoLayeredFFN> shared_mlp_;

  // FFN component (dense layers).
  std::shared_ptr<TwoLayeredFFN> mlp_;

  // Non-owning pointer to the MoE workspace buffer, owned by HyV3Model.
  TensorBuffer* moe_buffer_;
};

// hy_v3 MTP (NextN / Multi-Token Prediction) layer.
// Wraps a standard HyV3DecoderLayer with the MTP-specific input pipeline: enorm(embedding),
// hnorm(last-token hidden), concat, eh_proj (hidden*2 -> hidden), then the decoder.
class HyV3MtpLayer {
 public:
  HyV3MtpLayer(int layer_idx, LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config,
               std::shared_ptr<HyV3DecoderLayer> decoder_layer);
  ~HyV3MtpLayer() = default;

  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context);

 private:
  std::shared_ptr<Layernorm> enorm_;
  std::shared_ptr<Layernorm> hnorm_;
  std::shared_ptr<BaseLayer> concat_layer_;
  std::shared_ptr<Linear> eh_proj_;
  std::shared_ptr<BaseLayer> gather_layer_;
  std::shared_ptr<HyV3DecoderLayer> decoder_layer_;
  std::shared_ptr<TpCommunicator> tp_comm_;
};

class HyV3 : public ModelInterface {
 public:
  HyV3() {}
  ~HyV3() = default;

  Status GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) override;
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;
  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) override;

 private:
  TensorBuffer* moe_buffer_;
  int first_k_dense_replace_;

  std::map<int, std::shared_ptr<HyV3DecoderLayer>> decoder_layers_;
};

class HyV3Model : public CommonModel {
 public:
  HyV3Model(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
            std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
            std::shared_ptr<BaseModelCacheLayout> block_cache_layout);
  ~HyV3Model() = default;

 private:
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;
  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel::GetHiddenUnitBufferTensors;

 private:
  HyV3 hy_v3_;

  // MTP (NextN) support. One decoder layer per NextN step; the step-to-layer walk lives in
  // HyV3MtpDraftModel, which MtpDecoder drives.
  std::unordered_map<int, std::shared_ptr<HyV3MtpLayer>> nextn_layers_;

  // Next-step hidden buffer for MTP (owned by the model, injected into ForwardingBuffers).
  std::vector<Tensor> next_step_hidden_tensors_;
};

}  // namespace ksana_llm
