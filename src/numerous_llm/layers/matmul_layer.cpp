/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/matmul_layer.h"

namespace numerous_llm {
// kernel host代码代补充
void matmul(const Tensor& input_a, const Tensor& input_b, Tensor output, cudaStream_t stream) {}

Status MatMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  matmul(input_tensors[0], input_tensors[1], output_tensors[0], stream_);

  return Status();
}
}  // namespace numerous_llm
