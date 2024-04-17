/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/attention_layer.h"

namespace ksana_llm {
Status AttentionLayer::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, context, rank);
  int parameter_index = 0;
  layer_index_ = std::any_cast<const int>(parameters[parameter_index++]);
  int max_position_embeddings = std::any_cast<const int>(parameters[parameter_index++]);
  num_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  num_kv_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  head_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  stride_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  tensor_para_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  uint32_t rotary_dim = std::any_cast<const int>(parameters[parameter_index++]);
  float base = std::any_cast<const float>(parameters[parameter_index++]);
  bool is_neox = std::any_cast<const bool>(parameters[parameter_index++]);
  bool is_alibi = std::any_cast<const bool>(parameters[parameter_index++]);
  half* cos_sin_cache_ptr = std::any_cast<half*>(parameters[parameter_index++]);

  block_size_ = GetBlockManager()->GetBlockSize();
  block_token_num_ = GetBlockManager()->GetBlockTokenNum();

  // setting scaling factor and mode
  if (!is_alibi) {
    RoPEScalingFactor rope_scaling_factor_config =
        std::any_cast<const RoPEScalingFactor>(parameters[parameter_index++]);
    llm_kernels::nvidia::RotaryEmbeddingType rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::DEFAULT;
    float scaling_factor = 1.0f;
    if (rope_scaling_factor_config.type == "dynamic") {
      rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::DYNAMIC_NTK_SCALING;
      scaling_factor = rope_scaling_factor_config.factor;
    } else if (rope_scaling_factor_config.type != "default") {
      throw std::invalid_argument(fmt::format("Unsupport rope scaling type: {}", rope_scaling_factor_config.type));
    }

    rotary_embedding_cuda_.SetConfig(cos_sin_cache_ptr, rotary_dim, max_position_embeddings, base, head_size_,
                                     num_heads_, num_kv_heads_, stride_size_, is_neox,
                                     context_->GetComputeStreams()[rank_].Get(), rotary_embedding_type, scaling_factor);
  } else {
    llm_kernels::nvidia::GetAlibiSlopesCuda(reinterpret_cast<float*>(cos_sin_cache_ptr), num_heads_ * tensor_para_size_,
                                            context_->GetComputeStreams()[rank_].Get());
    alibi_slopes_ = reinterpret_cast<void*>(cos_sin_cache_ptr) + num_heads_ * rank_ * sizeof(float);
  }
  StreamSynchronize(context_->GetComputeStreams()[rank_]);
  return Status();
}

}  // namespace ksana_llm
