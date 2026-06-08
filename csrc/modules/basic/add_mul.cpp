/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/add_mul.h"

#include "layers/add_mul_layer.h"

namespace ksana_llm {

AddMul::AddMul(float scale1, float scale2, const LayerCreationContext& creation_context) {
  add_mul_layer_ = std::make_shared<AddMulLayer>();
  add_mul_layer_->Init({scale1, scale2}, creation_context.runtime_config, creation_context.context,
                       creation_context.rank);
}

AddMul::~AddMul() {}

Status AddMul::Forward(const Tensor* input1, const Tensor* input2, Tensor* output) {
  auto* typed = static_cast<AddMulLayer*>(add_mul_layer_.get());
  STATUS_CHECK_RETURN(typed->Forward(input1, input2, output));
  return Status();
}

}  // namespace ksana_llm
