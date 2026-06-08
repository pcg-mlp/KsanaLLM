/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/sigmoid.h"

#include "layers/activation_layer.h"

namespace ksana_llm {

Sigmoid::Sigmoid(const LayerCreationContext& creation_context) {
  sigmoid_layer_ = std::make_shared<SigmoidLayer>();
  sigmoid_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

Status Sigmoid::Forward(std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  auto* typed = static_cast<SigmoidLayer*>(sigmoid_layer_.get());
  STATUS_CHECK_RETURN(typed->Forward(&input_tensors[0]));
  output_tensors[0] = input_tensors[0];
  return Status();
}

}  // namespace ksana_llm
