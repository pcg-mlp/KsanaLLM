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

template <typename T>
Status AddLayer::ForwardT(const Tensor* a, const Tensor* b, Tensor* output) {
  const void* a_ptr = reinterpret_cast<const void*>(a->GetPtr<void>());
  const void* b_ptr = reinterpret_cast<const void*>(b->GetPtr<void>());
  if (a->shape[0] == b->shape[0]) {
    InvokeAddBiasResidual<T>(a_ptr, b_ptr, nullptr, static_cast<int>(a->shape[0]), static_cast<int>(a->shape[1]),
                             output->GetPtr<void>(), context_->device->GetPrimaryComputeStream().Get());
  } else {
    const int m = static_cast<int>(a->shape[0]);
    const int n = static_cast<int>(a->shape[1]);
    const int bias_stride = n * (m / static_cast<int>(b->shape[0]));
    InvokeAddBiasResidual<T>(a_ptr, nullptr, b_ptr, m, n, bias_stride, output->GetPtr<void>(),
                             context_->device->GetPrimaryComputeStream().Get());
  }
  output->shape = a->shape;
  output->dtype = a->dtype;
  return Status();
}

Status AddLayer::Forward(const Tensor* a, const Tensor* b, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, a, b, output);
}

}  // namespace ksana_llm
