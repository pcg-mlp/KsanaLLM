/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/attention_layer.h"

namespace ksana_llm {

template <typename T>
Status AttentionLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, context, rank);
  int parameter_index = 0;
  layer_index_ = std::any_cast<const int>(parameters[parameter_index++]);
  layer_num_ = std::any_cast<const int>(parameters[parameter_index++]);
  int max_position_embeddings = std::any_cast<const int>(parameters[parameter_index++]);
  num_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  num_kv_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  head_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  stride_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  tensor_para_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  kv_cache_dtype_ = std::any_cast<DataType>(parameters[parameter_index++]);
  k_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  v_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  uint32_t rotary_dim = std::any_cast<const int>(parameters[parameter_index++]);
  float base = std::any_cast<const float>(parameters[parameter_index++]);
  bool is_neox = std::any_cast<const bool>(parameters[parameter_index++]);
  PositionEncoding position_encoding = std::any_cast<const PositionEncoding>(parameters[parameter_index++]);
  void* cos_sin_cache_ptr = std::any_cast<void*>(parameters[parameter_index++]);

  block_size_ = GetBlockManager()->GetBlockSize();
  block_token_num_ = GetBlockManager()->GetBlockTokenNum();

  // setting scaling factor and mode
  if (position_encoding == PositionEncoding::ROPE) {
    RoPEScalingFactor rope_scaling_factor_config =
        std::any_cast<const RoPEScalingFactor>(parameters[parameter_index++]);
    llm_kernels::nvidia::RotaryEmbeddingType rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::DEFAULT;
    float scaling_factor = 1.0f;
    float low_freq_factor = 1.0f;
    float high_freq_factor = 4.0f;
    int original_max_position_embeddings = 8192;
    if (rope_scaling_factor_config.type == "dynamic") {
      rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::DYNAMIC_NTK_SCALING;
      scaling_factor = rope_scaling_factor_config.factor;
    } else if (rope_scaling_factor_config.type == "linear") {
      rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::LINEAR_SCALING;
      scaling_factor = rope_scaling_factor_config.factor;
    } else if (rope_scaling_factor_config.type == "llama3") {
      rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::MULTIFREQ_SCALING;
      scaling_factor = rope_scaling_factor_config.factor;
      low_freq_factor = rope_scaling_factor_config.low_freq_factor;
      high_freq_factor = rope_scaling_factor_config.high_freq_factor;
      original_max_position_embeddings = rope_scaling_factor_config.original_max_position_embeddings;
    } else if (rope_scaling_factor_config.type != "default") {
      KLLM_THROW(fmt::format("Unsupport rope scaling type: {}.", rope_scaling_factor_config.type));
    }

    rotary_embedding_cuda_.emplace();
    rotary_embedding_cuda_->SetConfig(static_cast<T*>(cos_sin_cache_ptr), rotary_dim, max_position_embeddings, base,
                                      head_size_, num_heads_, num_kv_heads_, stride_size_, is_neox,
                                      context_->GetComputeStreams()[rank_].Get(), rotary_embedding_type, scaling_factor,
                                      low_freq_factor, high_freq_factor, original_max_position_embeddings);
  } else if (position_encoding == PositionEncoding::ALIBI) {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::GetAlibiSlopesCuda(reinterpret_cast<float*>(cos_sin_cache_ptr),
                                                                  num_heads_ * tensor_para_size_,
                                                                  context_->GetComputeStreams()[rank_].Get()));
    alibi_slopes_ = reinterpret_cast<void*>(cos_sin_cache_ptr) + num_heads_ * rank_ * sizeof(float);
  }
  StreamSynchronize(context_->GetComputeStreams()[rank_]);
  return Status();
}

template class AttentionLayer<float>;
template class AttentionLayer<half>;
#ifdef ENABLE_BFLOAT16
template class AttentionLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
