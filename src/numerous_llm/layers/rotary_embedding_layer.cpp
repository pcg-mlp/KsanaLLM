/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/rotary_embedding_layer.h"

namespace numerous_llm {
// kernel host代码代补充
void rotary_embedding(const Tensor& input, Tensor output, int max_position, cudaStream_t stream) {}

// TODO: 析构释放

Status RotaryEmbeddingLayer::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  int parameter_index = 0;
  int max_position_embeddings = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("max_position_embeddings {}", max_position_embeddings);

  int rotary_dim = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("rotary_dim {}", rotary_dim);

  float base = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("base {}", base);

  int head_size = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("head_size {}", head_size);

  int num_heads = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("num_heads {}", num_heads);

  int num_kv_heads = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("num_kv_heads {}", num_kv_heads);

  bool is_neox = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("is_neox {}", is_neox);

  // 创建 cos_sin_cache 空间
  size_t total_bytes = rotary_dim * max_position_embeddings * sizeof(half);
  GetBlockManager()->SetDeviceId(rank);
  GetBlockManager()->AllocateContiguous(total_bytes, cos_sin_cache_block_id_);

  //rotary_embedding_cuda_.SetConfig(GetBlockPtrs<half>(cos_sin_cache_block_id)[0], rotary_dim, max_position_embeddings,
  //                                 base, head_size, num_heads, num_kv_heads, is_neox,
  //                                 context_->GetComputeStreams()[rank]);
  return Status();
}

Status RotaryEmbeddingLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  rotary_embedding(input_tensors[0], output_tensors[0], max_position_embeddings_, context_->GetComputeStreams()[rank_]);
  return Status();
}

}  // namespace numerous_llm
