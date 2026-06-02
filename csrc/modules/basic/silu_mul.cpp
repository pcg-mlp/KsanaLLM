/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/silu_mul.h"

#include "layers/silu_mul_layer.h"

namespace ksana_llm {

SiluMul::SiluMul(const LayerCreationContext& creation_context) {
  silu_mul_layer_ = std::make_shared<SiluMulLayer>();
  silu_mul_layer_->Init({creation_context.model_config.swiglu_limit}, creation_context.runtime_config,
                        creation_context.context, creation_context.rank);
}

SiluMul::~SiluMul() {}

Status SiluMul::Forward(Tensor bias, Tensor gated_bias, std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx) {
  STATUS_CHECK_RETURN(silu_mul_layer_->Forward({bias, gated_bias}, output_tensors, exec_ctx));
  return Status();
}

Status SiluMul::Forward(Tensor fused_tensor, std::vector<Tensor>& output_tensors, const LayerExecContext& exec_ctx) {
  STATUS_CHECK_RETURN(silu_mul_layer_->Forward({fused_tensor}, output_tensors, exec_ctx));
  return Status();
}

}  // namespace ksana_llm
