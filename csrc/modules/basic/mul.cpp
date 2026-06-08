/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/mul.h"

#include "layers/mul_layer.h"

namespace ksana_llm {

Mul::Mul(const LayerCreationContext& creation_context) {
  mul_layer_ = std::make_shared<MulLayer>();
  mul_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

Status Mul::Forward(Tensor A, Tensor B, std::vector<Tensor>& output_tensors) {
  auto* typed = static_cast<MulLayer*>(mul_layer_.get());
  STATUS_CHECK_RETURN(typed->Forward(&A, &B, &output_tensors[0]));
  return Status();
}

}  // namespace ksana_llm
