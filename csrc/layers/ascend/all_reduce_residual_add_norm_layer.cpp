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

Status AllReduceResidualAddNormLayer::Forward(const std::vector<Tensor>& input_tensors,
                                              std::vector<Tensor>& output_tensors) {
  return Status(RET_NOT_IMPLEMENTED, "AllReduceResidualAddNormLayer overloaded forward is not implemented");
}

}  // namespace ksana_llm
