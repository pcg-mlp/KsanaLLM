/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/add_layer.h"

namespace numerous_llm {
// kernel host代码代补充
void add(const Tensor& input_a, const Tensor& input_b, Tensor output, cudaStream_t stream) {}

Status AddLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  add(input_tensors[0], input_tensors[1], output_tensors[0], context_->GetComputeStreams()[rank_]);
  return Status();
}
}  // namespace numerous_llm
