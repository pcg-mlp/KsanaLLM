/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cstdlib>
#include "layers/assemble_tokens_hidden_layer.h"

#include "device/common_device.h"
#include "device/device_utils.h"

namespace ksana_llm {

Status AssembleTokensHiddenLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                       std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "AssembleTokensHiddenLayer not supported.");
}

template <typename T>
Status AssembleTokensHiddenLayer::ForwardT(const Tensor& hidden_states, const Tensor& accepted_token_indices,
                                           Tensor& output) {
  return Status(RET_UNDEFINED_REFERENCE, "AssembleTokensHiddenLayer not supported.");
}

Status AssembleTokensHiddenLayer::Forward(const Tensor* hidden_states, const Tensor* accepted_token_indices,
                                          Tensor* output, const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "AssembleTokensHiddenLayer not supported.");
}

}  // namespace ksana_llm
