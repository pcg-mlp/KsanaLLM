/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/activation_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

void InvokeSiluActivation(const void* input, const void* bias, const int m, const int n, void* output,
                          cudaStream_t stream);
Status ActivationLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  InvokeSiluActivation(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
                       reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>()),
                       static_cast<int>(input_tensors[0].shape[0]), static_cast<int>(input_tensors[0].shape[1]),
                       output_tensors[0].GetPtr<void>(), context_->GetComputeStreams()[rank_]);

  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}
}  // namespace ksana_llm
