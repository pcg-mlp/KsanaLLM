/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/situ_mul.h"

#include "layers/situ_mul_layer.h"

namespace ksana_llm {

SituMul::SituMul(const LayerCreationContext& creation_context) {
  situ_mul_layer_ = std::make_shared<SituMulLayer>();
  SituMulLayerParameters params{
      .situ_beta = creation_context.model_config.situ_beta,
      .situ_linear_beta = creation_context.model_config.situ_linear_beta,
  };
  situ_mul_layer_->Init({params}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

SituMul::~SituMul() {}

Status SituMul::Forward(Tensor fused_tensor, std::vector<Tensor>& output_tensors, const LayerExecContext& exec_ctx) {
  auto* typed = static_cast<SituMulLayer*>(situ_mul_layer_.get());
  STATUS_CHECK_RETURN(typed->Forward(&fused_tensor, &output_tensors[0], exec_ctx));
  return Status();
}

}  // namespace ksana_llm
