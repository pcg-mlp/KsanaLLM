/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/add_mul_layer.h"

#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status AddMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                         std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);

  // First parameter: scale1
  if (parameters.size() > 0) {
    scale1_ = std::any_cast<float>(parameters[0]);
  }

  // Second parameter: scale2
  if (parameters.size() > 1) {
    scale2_ = std::any_cast<float>(parameters[1]);
  }

  return Status();
}

template <typename T>
Status AddMulLayer::ForwardT(const Tensor& input1, const Tensor& input2, Tensor& output) {
  output.shape = input1.shape;
  output.dtype = input1.dtype;

  const int m = static_cast<int>(input1.shape[0]);
  const int n = static_cast<int>(input1.shape[1]);
  cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

  InvokeMulThenAdd<T>(reinterpret_cast<const void*>(input1.GetPtr<void>()),
                      reinterpret_cast<const void*>(input2.GetPtr<void>()), scale1_, scale2_, m, n,
                      output.GetPtr<void>(), stream);

  return Status();
}

Status AddMulLayer::Forward(const Tensor* input1, const Tensor* input2, Tensor* output,
                            const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *input1, *input2, *output);
}

}  // namespace ksana_llm
