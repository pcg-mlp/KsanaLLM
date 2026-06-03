/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/add_layer.h"

#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status AddLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

Status AddLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status AddLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  auto a = reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>());
  auto b = reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>());
  const int m = static_cast<int>(input_tensors[0].shape[0]);
  const int n = static_cast<int>(input_tensors[0].shape[1]);
  // The Add-Bias-Residual Kernel uses the shape[0] of the input tensor to determine whether
  // broadcasting is required.
  if (input_tensors[0].shape[0] == input_tensors[1].shape[0]) {
    InvokeAddBiasResidual<T>(a, b, nullptr, m, n, output_tensors[0].GetPtr<void>(),
                             context_->device->GetPrimaryComputeStream().Get());
  } else {
    const int bias_stride = n * (m / static_cast<int>(input_tensors[1].shape[0]));
    InvokeAddBiasResidual<T>(a, nullptr, b, m, n, bias_stride, output_tensors[0].GetPtr<void>(),
                             context_->device->GetPrimaryComputeStream().Get());
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace ksana_llm
