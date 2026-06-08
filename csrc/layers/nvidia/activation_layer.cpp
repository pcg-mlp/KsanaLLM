/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/activation_layer.h"

#include "csrc/kernels/nvidia/activation/activation.h"
#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

bool IsGatedActivation(ActivationType type) {
  if (type == ActivationType::Geglu || type == ActivationType::Swiglu) {
    return true;
  }
  return false;
}

Status ActivationLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                             std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  activation_type_ = std::any_cast<const ActivationType>(parameters[0]);
  return Status();
}

template <typename T>
Status ActivationLayer::ForwardT(const Tensor* input, Tensor* output, const Tensor* gated_weight, const Tensor* bias,
                                 const Tensor* gated_bias) {
  const void* gated_weight_ptr = (gated_weight != nullptr && IsGatedActivation(activation_type_))
                                     ? reinterpret_cast<const void*>(gated_weight->GetPtr<void>())
                                     : nullptr;
  const void* bias_ptr = bias != nullptr ? reinterpret_cast<const void*>(bias->GetPtr<void>()) : nullptr;
  const void* gated_bias_ptr = (gated_bias != nullptr && IsGatedActivation(activation_type_))
                                   ? reinterpret_cast<const void*>(gated_bias->GetPtr<void>())
                                   : nullptr;

  if (activation_type_ == ActivationType::Gelu || activation_type_ == ActivationType::Geglu) {
    InvokeGatedActivation<llm_kernels::nvidia::GeluActivation, T>(
        reinterpret_cast<const void*>(input->GetPtr<void>()), bias_ptr, gated_weight_ptr, gated_bias_ptr,
        static_cast<int>(input->shape[0]), static_cast<int>(input->shape[1]), output->GetPtr<void>(),
        context_->device->GetPrimaryComputeStream().Get());
  } else if (activation_type_ == ActivationType::Relu) {
    InvokeGatedActivation<llm_kernels::nvidia::ReluActivation, T>(
        reinterpret_cast<const void*>(input->GetPtr<void>()), bias_ptr, gated_weight_ptr, gated_bias_ptr,
        static_cast<int>(input->shape[0]), static_cast<int>(input->shape[1]), output->GetPtr<void>(),
        context_->device->GetPrimaryComputeStream().Get());
  } else {  // activation_type_ == ActivationType::Swiglu
    InvokeGatedActivation<llm_kernels::nvidia::SiluActivation, T>(
        reinterpret_cast<const void*>(input->GetPtr<void>()), bias_ptr, gated_weight_ptr, gated_bias_ptr,
        static_cast<int>(input->shape[0]), static_cast<int>(input->shape[1]), output->GetPtr<void>(),
        context_->device->GetPrimaryComputeStream().Get());
  }
  output->shape = input->shape;
  output->dtype = input->dtype;
  return Status();
}

Status ActivationLayer::Forward(const Tensor* input, Tensor* output, const Tensor* gated_weight, const Tensor* bias,
                                const Tensor* gated_bias, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input, output, gated_weight, bias, gated_bias);
}

template <typename T>
Status SigmoidLayer::ForwardT(Tensor* inout) {
  size_t size = inout->shape[0] * inout->shape[1];
  float scale = 1.0f;
  InvokeSigmoidActivation<T>(inout->GetPtr<void>(), size, scale, context_->device->GetPrimaryComputeStream().Get());
  return Status();
}

Status SigmoidLayer::Forward(Tensor* inout, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, inout);
}

}  // namespace ksana_llm
