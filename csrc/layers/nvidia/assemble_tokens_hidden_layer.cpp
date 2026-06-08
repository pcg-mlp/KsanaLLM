/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/assemble_tokens_hidden_layer.h"

#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status AssembleTokensHiddenLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                       std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

template <typename T>
Status AssembleTokensHiddenLayer::ForwardT(const Tensor& hidden_states, const Tensor& accepted_token_indices,
                                           Tensor& output) {
  const int accepted_tokens_size = accepted_token_indices.shape[0];
  AssembleTokensHidden<T>(reinterpret_cast<const void*>(hidden_states.GetPtr<void>()),
                          reinterpret_cast<const void*>(accepted_token_indices.GetPtr<void>()), accepted_tokens_size,
                          hidden_states.shape[1], reinterpret_cast<void*>(output.GetPtr<void>()),
                          context_->device->GetPrimaryComputeStream().Get());
  output.shape = hidden_states.shape;
  output.shape[0] = accepted_tokens_size;
  output.dtype = hidden_states.dtype;
  return Status();
}

Status AssembleTokensHiddenLayer::Forward(const Tensor* hidden_states, const Tensor* accepted_token_indices,
                                          Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *hidden_states, *accepted_token_indices, *output);
}

}  // namespace ksana_llm
