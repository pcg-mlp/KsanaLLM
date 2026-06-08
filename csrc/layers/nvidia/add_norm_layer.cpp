/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/add_norm_layer.h"
#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status AddNormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  KLLM_LOG_DEBUG << fmt::format("rms_norm_eps {}", rms_norm_eps_);
  return Status();
}

template <typename T>
Status AddNormLayer::ForwardT(const Tensor* input, const Tensor* residual, const Tensor* weight, Tensor* output) {
  if (input->GetPtr<void>() != output->GetPtr<void>()) {
    KLLM_THROW("AddNormLayer input and output tensors must be the same because it is an inplace operation.");
  }
  // Step 1: residual[i] += input[i]
  // Step 2: input[i] = (residual[i] / RMS(residual)) * weight[i]
  InvokeFusedAddRmsNorm<T>(input->GetPtr<void>(), residual->GetPtr<void>(), weight->GetPtr<void>(), rms_norm_eps_,
                           input->shape[0], input->shape[1], context_->device->GetPrimaryComputeStream().Get());
  output->shape = input->shape;
  output->dtype = input->dtype;
  return Status();
}

Status AddNormLayer::Forward(const Tensor* input, const Tensor* residual, const Tensor* weight, Tensor* output,
                             const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input, residual, weight, output);
}

}  // namespace ksana_llm
