/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "models/base/layer_creation_context.h"

namespace ksana_llm {

class GatherFirstDim {
 public:
  // Disable a default constructor
  explicit GatherFirstDim(const LayerCreationContext& creation_context);

  ~GatherFirstDim() = default;

  Status Forward(const Tensor& data, const Tensor& indices, Tensor& output);

 protected:
  std::unique_ptr<BaseLayer> gather_first_dim_layer_;
};

}  // namespace ksana_llm
