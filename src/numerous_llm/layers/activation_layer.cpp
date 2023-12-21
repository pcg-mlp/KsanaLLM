/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/activation_layer.h"

namespace numerous_llm {
// kernel host代码代补充
void activation(const Tensor& input, Tensor output, cudaStream_t stream) {}

Status ActivationLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  activation(input_tensors[0], output_tensors[0], stream_);
  return Status();
}
}  // namespace numerous_llm
