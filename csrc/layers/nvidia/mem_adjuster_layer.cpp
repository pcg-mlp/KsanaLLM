/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/mem_adjuster_layer.h"

#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status MemAdjusterLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                              std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

Status MemAdjusterLayer::ExtractSubMatrix(const Tensor& input_tensor, Tensor& output_tensor, size_t input_offset,
                                          size_t output_n) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ExtractSubMatrixT, input_tensor, output_tensor, input_offset, output_n);
}

template <typename T>
Status MemAdjusterLayer::ExtractSubMatrixT(const Tensor& input_tensor, Tensor& output_tensor, size_t input_offset,
                                           size_t output_n) {
  InvokeExtractSubMatrix(input_tensor.GetPtr<T>() + input_offset, output_tensor.GetPtr<T>(), input_tensor.shape[0],
                         input_tensor.shape[1], output_n, context_->device->GetPrimaryComputeStream().Get());
  output_tensor.shape[0] = input_tensor.shape[0];
  output_tensor.shape[1] = output_n;
  return Status();
}

}  // namespace ksana_llm
