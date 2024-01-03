/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/silu_mul_layer.h"
#include "numerous_llm/kernels/nvidia/kernel_wrapper.h"

namespace numerous_llm {
// kernel host代码代补充
void silu_mul(const Tensor& input_a, const Tensor& input_b, Tensor output, cudaStream_t stream) {}

Status SiluMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>());
  InvokeSiluActivation(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
                        reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>()),
                        static_cast<int>(input_tensors[0].shape[0]), static_cast<int>(input_tensors[0].shape[1]),
                        output_tensors[0].GetPtr<void>(), context_->GetComputeStreams()[rank_]);
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}
}  // namespace numerous_llm
