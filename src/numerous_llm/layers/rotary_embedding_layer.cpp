/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/rotary_embedding_layer.h"

namespace numerous_llm {
// kernel host代码代补充
void rotary_embedding(const Tensor& input, Tensor output, int max_position, cudaStream_t stream) {}

Status RotaryEmbeddingLayer::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  int parameter_index = 0;
  max_position_embeddings_ = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("max_position_embeddings {}", max_position_embeddings_);
  return Status();
}

Status RotaryEmbeddingLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  rotary_embedding(input_tensors[0], output_tensors[0], max_position_embeddings_, context_->GetComputeStreams()[rank_]);
  return Status();
}

}  // namespace numerous_llm
