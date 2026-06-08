/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/silu_mul_layer.h"

namespace ksana_llm {

Status SiluMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status(RET_UNDEFINED_REFERENCE, "SiluMulLayer not supported.");
}

template <typename T>
Status SiluMulLayer::ForwardFusedT(const Tensor& fused_gate_up, Tensor& output, const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "SiluMulLayer not supported.");
}

template <typename T>
Status SiluMulLayer::ForwardSplitT(const Tensor& gate, const Tensor& up, Tensor& output,
                                   const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "SiluMulLayer not supported.");
}

Status SiluMulLayer::Forward(const Tensor* fused_gate_up, Tensor* output, const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "SiluMulLayer not supported.");
}

Status SiluMulLayer::Forward(const Tensor* gate, const Tensor* up, Tensor* output, const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "SiluMulLayer not supported.");
}

}  // namespace ksana_llm
