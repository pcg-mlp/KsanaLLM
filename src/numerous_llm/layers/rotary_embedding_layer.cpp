/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/rotary_embedding_layer.h"

namespace numerous_llm {

Status RotaryEmbeddingLayer::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  int parameter_index = 0;
  int max_position_embeddings = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("max_position_embeddings {}", max_position_embeddings);

  uint32_t rotary_dim = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("rotary_dim {}", rotary_dim);

  float base = std::any_cast<const float>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("base {}", base);

  uint32_t head_size = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("head_size {}", head_size);

  size_t num_heads = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("num_heads {}", num_heads);

  size_t num_kv_heads = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("num_kv_heads {}", num_kv_heads);

  bool is_neox = std::any_cast<const bool>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("is_neox {}", is_neox);

  // 创建 cos_sin_cache 空间
  size_t total_bytes = rotary_dim * max_position_embeddings * sizeof(half);
  GetBlockManager()->SetDeviceId(rank_);
  GetBlockManager()->AllocateContiguous(total_bytes, cos_sin_cache_block_id_);
  void* cos_sin_cache_ptr;
  GetBlockManager()->GetContiguousPtr(cos_sin_cache_block_id_, cos_sin_cache_ptr);
  CUDA_CHECK(cudaStreamSynchronize(context_->GetMemoryManageStreams()[rank_]));
  rotary_embedding_cuda_.SetConfig(reinterpret_cast<half*>(cos_sin_cache_ptr), rotary_dim, max_position_embeddings,
                                   base, head_size, num_heads, num_kv_heads, is_neox,
                                   context_->GetMemoryManageStreams()[rank_]);
  CUDA_CHECK(cudaStreamSynchronize(context_->GetMemoryManageStreams()[rank_]));
  return Status();
}

Status RotaryEmbeddingLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  size_t qkv_size = input_tensors[0].GetTotalBytes();
  NLLM_LOG_INFO << fmt::format("qkv bytes size = {}", qkv_size);
  void* qkv_ptr = input_tensors[0].GetPtr<void>();

  void* q_ptr = qkv_ptr;
  void* k_ptr = qkv_ptr + qkv_size / 3;

  int num_tokens = input_tensors[2].shape[0];

  rotary_embedding_cuda_.SetInput(input_tensors[1].GetPtr<int64_t>(), static_cast<half*>(q_ptr),
                                  static_cast<half*>(k_ptr), num_tokens, context_->GetComputeStreams()[rank_]);
  rotary_embedding_cuda_.Forward();
  return Status();
}

}  // namespace numerous_llm
