/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/add_norm_layer.h"

namespace ksana_llm {

Status AddNormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "AddNormLayer not supported.");
}

template <typename T>
Status AddNormLayer::ForwardT(const Tensor* input, const Tensor* residual, const Tensor* weight, Tensor* output) {
  return Status(RET_UNDEFINED_REFERENCE, "AddNormLayer not supported.");
}

Status AddNormLayer::Forward(const Tensor* input, const Tensor* residual, const Tensor* weight, Tensor* output,
                             const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "AddNormLayer not supported.");
}

}  // namespace ksana_llm
