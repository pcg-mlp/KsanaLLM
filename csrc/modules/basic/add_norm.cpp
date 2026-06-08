/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/add_norm.h"
#include "layers/add_norm_layer.h"
namespace ksana_llm {

AddNorm::AddNorm(const std::string& weight_name, float norm_eps, const LayerCreationContext& creation_context) {
  add_norm_layer_ = std::make_shared<AddNormLayer>();
  weight_ = creation_context.base_weight->GetModelWeights(weight_name);
  add_norm_layer_->Init({norm_eps}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

AddNorm::~AddNorm() {}

Status AddNorm::Forward(const Tensor* input, const Tensor* residual, Tensor* output) {
  auto* typed = static_cast<AddNormLayer*>(add_norm_layer_.get());
  STATUS_CHECK_RETURN(typed->Forward(input, residual, &weight_, output));
  return Status();
}

}  // namespace ksana_llm
