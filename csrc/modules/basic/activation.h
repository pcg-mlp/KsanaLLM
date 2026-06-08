/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "models/base/layer_creation_context.h"

namespace ksana_llm {

class Activation {
 public:
  // Disable a default constructor
  explicit Activation(const std::string& activation_type, const LayerCreationContext& creation_context);

  ~Activation();

  Status Forward(const Tensor* input, Tensor* output, const Tensor* gated = nullptr, const Tensor* bias = nullptr,
                 const Tensor* gated_bias = nullptr);

 protected:
  std::shared_ptr<BaseLayer> activation_layer_;
};
}  // namespace ksana_llm
