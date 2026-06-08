/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/add_norm_layer.h"

namespace ksana_llm {

Status AddNormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context> context, int rank) {
  int parameter_index = 0;
  auto layernorm_eps = std::any_cast<float>(parameters[parameter_index++]);
  layernorm_layer_ = std::make_shared<LayernormLayer>();
  layernorm_layer_->Init({layernorm_eps}, runtime_config, context, rank);
  add_layer_ = std::make_shared<AddLayer>();
  add_layer_->Init({}, runtime_config, context, rank);
  return Status();
}

Status AddNormLayer::Forward(const Tensor* input, const Tensor* residual, const Tensor* weight, Tensor* output,
                             const LayerExecContext& exec_ctx) {
  Tensor add_result = *residual;
  STATUS_CHECK_RETURN(add_layer_->Forward(input, residual, &add_result));
  STATUS_CHECK_RETURN(layernorm_layer_->Forward(residual, weight, output));
  return Status();
}

}  // namespace ksana_llm
