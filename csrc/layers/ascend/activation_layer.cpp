/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/activation_layer.h"

namespace ksana_llm {

Status ActivationLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                             std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "ActivationLayer not supported.");
}

Status ActivationLayer::Forward(const Tensor* input, Tensor* output, const Tensor* gated_weight, const Tensor* bias,
                                const Tensor* gated_bias, const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "ActivationLayer not supported.");
}

Status SigmoidLayer::Forward(Tensor* inout, const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "SigmoidLayer not supported.");
}

}  // namespace ksana_llm
