/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/gather_first_dim.h"

#include "layers/gather_first_dim_layer.h"

namespace ksana_llm {

GatherFirstDim::GatherFirstDim(const LayerCreationContext& creation_context) {
  gather_first_dim_layer_ = std::make_unique<GatherFirstDimLayer>();
  gather_first_dim_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

Status GatherFirstDim::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return gather_first_dim_layer_->Forward(input_tensors, output_tensors);
}

}  // namespace ksana_llm
