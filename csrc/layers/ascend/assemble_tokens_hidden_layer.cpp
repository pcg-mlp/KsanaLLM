/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/assemble_tokens_hidden_layer.h"

#include <cstdlib>

#include "device/common_device.h"
#include "device/device_utils.h"
#include "kernels/ascend/kernel_wrapper.h"

namespace ksana_llm {

Status AssembleTokensHiddenLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                       std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  context_ = context;
  rank_ = rank;
  atb::infer::GatherParam gather_param;
  atb_op_executor_.Init(rank, gather_param);
  return Status();
}

template <typename T>
Status AssembleTokensHiddenLayer::ForwardT(const Tensor& hidden_states, const Tensor& accepted_token_indices,
                                           Tensor& output) {
  output.shape = hidden_states.shape;
  output.shape[0] = accepted_token_indices.shape[0];
  output.dtype = hidden_states.dtype;
  reinterpret_cast<atb::Context*>(GetRuntimeContext())->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
  atb_op_executor_.ResetVariantPack();
  atb_op_executor_.SetInputTensor(hidden_states.GetPtr<void>(), hidden_states.shape,
                                  static_cast<aclDataType>(DataType(hidden_states.dtype)));
  atb_op_executor_.SetInputTensor(accepted_token_indices.GetPtr<void>(), accepted_token_indices.shape,
                                  static_cast<aclDataType>(DataType(accepted_token_indices.dtype)));
  atb_op_executor_.SetOutputTensor(output.GetPtr<void>(), output.shape,
                                   static_cast<aclDataType>(DataType(output.dtype)));
  atb_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext()), GetWorkSpaceFunc());
  return Status();
}

Status AssembleTokensHiddenLayer::Forward(const Tensor* hidden_states, const Tensor* accepted_token_indices,
                                          Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *hidden_states, *accepted_token_indices, *output);
}

}  // namespace ksana_llm
