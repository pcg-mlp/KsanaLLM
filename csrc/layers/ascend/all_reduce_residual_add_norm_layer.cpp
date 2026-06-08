/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/all_reduce_residual_add_norm_layer.h"

#include "models/communicator/tp_communicator.h"

namespace ksana_llm {

Status AllReduceResidualAddNormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                           std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  rms_norm_weight_ = std::any_cast<Tensor>(parameters[parameter_index++]);
  tp_comm_ = std::any_cast<std::shared_ptr<TpCommunicator>>(parameters[parameter_index++]);
  return Status();
}

template <typename T>
Status AllReduceResidualAddNormLayer::ForwardT(const Tensor& input, const Tensor* residual_in, Tensor& output) {
  return Status(RET_NOT_IMPLEMENTED, "AllReduceResidualAddNormLayer is not implemented on Ascend.");
}

Status AllReduceResidualAddNormLayer::Forward(const Tensor* input, Tensor* residual_inout,
                                              const LayerExecContext& exec_ctx) {
  return Status(RET_NOT_IMPLEMENTED, "AllReduceResidualAddNormLayer is not implemented on Ascend.");
}

Status AllReduceResidualAddNormLayer::Forward(const Tensor* input, const Tensor* residual, Tensor* output,
                                              const LayerExecContext& exec_ctx) {
  return Status(RET_NOT_IMPLEMENTED, "AllReduceResidualAddNormLayer is not implemented on Ascend.");
}

}  // namespace ksana_llm
