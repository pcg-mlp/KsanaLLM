/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/attention_layer.h"

namespace ksana_llm {
Status AttentionLayer::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, context, rank);
  int parameter_index = 0;
  layer_index_ = std::any_cast<const int>(parameters[parameter_index++]);
  max_position_embeddings_ = std::any_cast<const int>(parameters[parameter_index++]);
  num_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  num_kv_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  head_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  uint32_t rotary_dim = std::any_cast<const int>(parameters[parameter_index++]);
  float base = std::any_cast<const float>(parameters[parameter_index++]);
  bool is_neox = std::any_cast<const bool>(parameters[parameter_index++]);
  size_t total_bytes = rotary_dim * max_position_embeddings_ * sizeof(half);
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->AllocateContiguous(total_bytes, cos_sin_cache_block_id_);
  // TODO: 完全一致是否存储一份
  void* cos_sin_cache_ptr;
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->GetContiguousPtr(cos_sin_cache_block_id_, cos_sin_cache_ptr);
  CUDA_CHECK(cudaStreamSynchronize(context_->GetMemoryManageStreams()[rank_]));
  rotary_embedding_cuda_.SetConfig(reinterpret_cast<half*>(cos_sin_cache_ptr), rotary_dim, max_position_embeddings_,
                                   base, head_size_, num_heads_, num_kv_heads_, is_neox,
                                   context_->GetMemoryManageStreams()[rank_]);
  CUDA_CHECK(cudaStreamSynchronize(context_->GetMemoryManageStreams()[rank_]));

  block_size_ = GetBlockManager()->GetBlockSize();
  block_token_num_ = GetBlockManager()->GetBlockTokenNum();

  NLLM_LOG_DEBUG << fmt::format("layer_index_ {}; max_position_embeddings {}; block_size_ {}; block_token_num_ {}",
                                layer_index_, max_position_embeddings_, block_size_, block_token_num_);
  return Status();
}

}  // namespace ksana_llm
