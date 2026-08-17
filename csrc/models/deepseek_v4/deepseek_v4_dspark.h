/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "layers/dspark_layer.h"
#include "models/common/common_model.h"
#include "modules/basic/layernorm.h"
#include "modules/basic/linear.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

class DeepSeekV4DecoderLayer;

class DeepSeekV4DSparkLayer {
 public:
  DeepSeekV4DSparkLayer(const int layer_idx, LayerCreationContext& creation_context,
                        ModelCreationConfig& model_creation_config,
                        std::shared_ptr<DeepSeekV4DecoderLayer> decoder_layer, bool is_embed_layer,
                        Tensor* dspark_hidden_buffer = nullptr);

  ~DeepSeekV4DSparkLayer() = default;

  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context);

  // Runs main_proj + main_norm (embed layer only), then wkv_proj + fused RMSNorm/RoPE/paged-store.
  Status InjectKVForward(ForwardingContext& forwarding_context);

 private:
  Status InjectKV(Tensor& kv_buffer, ForwardingContext& forwarding_context);

  const int layer_idx_;
  const int local_layer_idx_;
  const size_t hidden_dim_;
  const float layernorm_eps_;
  const DataType kv_cache_dtype_;

  // Inject-KV state (RMSNorm + RoPE + FP8 paged store).
  uint32_t page_bits_;
  void* cos_sin_cache_ptr_ = nullptr;

  // Only for layers require embedding.
  size_t num_target_layers_ = 0;
  std::unique_ptr<Linear> main_proj_;  // [num_target*hidden_dim -> hidden_dim]
  std::unique_ptr<Layernorm> main_norm_;

  std::unique_ptr<Linear> wkv_proj_;  // [hidden_dim -> head_dim]
  Tensor kv_norm_weight_;             // [head_dim]

  std::unique_ptr<DSparkLayer> dspark_layer_;
  std::shared_ptr<DeepSeekV4DecoderLayer> decoder_layer_;

  // Buffer for captured cross-layer hidden states (owned by DeepSeekV4Model).
  Tensor* dspark_hidden_buffer_ptr_ = nullptr;
};

}  // namespace ksana_llm
