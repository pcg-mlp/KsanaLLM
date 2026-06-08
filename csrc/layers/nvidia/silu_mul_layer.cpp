/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/silu_mul_layer.h"

#include "csrc/kernels/nvidia/activation/activation.h"
#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status SiluMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  swiglu_limit_ = std::any_cast<float>(parameters[parameter_index++]);
  return Status();
}

template <typename T>
Status SiluMulLayer::ForwardFusedT(const Tensor& fused_gate_up, Tensor& output, const LayerExecContext& exec_ctx) {
  InvokeRowBasedGatedActivation<llm_kernels::nvidia::SiluActivation, T>(
      reinterpret_cast<const void*>(fused_gate_up.GetPtr<void>()), static_cast<int>(fused_gate_up.shape[0]),
      static_cast<int>(fused_gate_up.shape[1]), output.GetPtr<void>(), GetComputeStream(exec_ctx), swiglu_limit_);
  output.shape = fused_gate_up.shape;
  output.shape[1] = output.shape[1] / 2;
  output.dtype = fused_gate_up.dtype;
  return Status();
}

template <typename T>
Status SiluMulLayer::ForwardSplitT(const Tensor& gate, const Tensor& up, Tensor& output,
                                   const LayerExecContext& exec_ctx) {
  InvokeGatedActivation<llm_kernels::nvidia::SiluActivation, T>(
      reinterpret_cast<const void*>(gate.GetPtr<void>()), /* bias */ nullptr,
      reinterpret_cast<const void*>(up.GetPtr<void>()), /* gated_bias */ nullptr, static_cast<int>(gate.shape[0]),
      static_cast<int>(gate.shape[1]), output.GetPtr<void>(), GetComputeStream(exec_ctx));
  output.shape = gate.shape;
  output.dtype = gate.dtype;
  return Status();
}

Status SiluMulLayer::Forward(const Tensor* fused_gate_up, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardFusedT, *fused_gate_up, *output, exec_ctx);
}

Status SiluMulLayer::Forward(const Tensor* gate, const Tensor* up, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardSplitT, *gate, *up, *output, exec_ctx);
}

}  // namespace ksana_llm
