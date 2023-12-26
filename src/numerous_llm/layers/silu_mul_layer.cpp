/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/silu_mul_layer.h"

namespace numerous_llm {
// kernel host代码代补充
void silu_mul(const Tensor& input_a, const Tensor& input_b, Tensor output, cudaStream_t stream) {}

Status SiluMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status();
}
}  // namespace numerous_llm
