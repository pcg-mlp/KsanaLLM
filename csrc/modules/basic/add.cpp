/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/add.h"

#include "layers/add_layer.h"

namespace ksana_llm {

Add::Add(const LayerCreationContext& creation_context, const std::string& weight_name) {
  add_layer_ = std::make_shared<AddLayer>();
  add_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
  if (weight_name != "") {
    weight_ = creation_context.base_weight->GetModelWeights(weight_name);
    with_weight_ = true;
  }
}

Add::~Add() {}

Status Add::Forward(Tensor A, Tensor B, std::vector<Tensor>& output_tensors) {
  auto* typed_layer = static_cast<AddLayer*>(add_layer_.get());
  STATUS_CHECK_RETURN(typed_layer->Forward(&A, &B, &output_tensors[0]));
  return Status();
}

Status Add::Forward(Tensor A, std::vector<Tensor>& output_tensors) {
  auto* typed_layer = static_cast<AddLayer*>(add_layer_.get());
  STATUS_CHECK_RETURN(typed_layer->Forward(&A, &weight_, &output_tensors[0]));
  return Status();
}

}  // namespace ksana_llm
