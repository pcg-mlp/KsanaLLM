/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "models/base/layer_creation_context.h"

namespace ksana_llm {

// Fused Add and RMS Norm operation (currently Layernorm is not supported)
class AddNorm {
 public:
  // Disable a default constructor
  AddNorm(const std::string& weight_name, float norm_eps, const LayerCreationContext& creation_context);

  ~AddNorm();

  Status Forward(const Tensor* input, const Tensor* residual, Tensor* output);

 protected:
  std::shared_ptr<BaseLayer> add_norm_layer_;
  Tensor weight_;
};
}  // namespace ksana_llm
