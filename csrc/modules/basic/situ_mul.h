/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "layers/layer_exec_context.h"
#include "models/base/layer_creation_context.h"

namespace ksana_llm {

class SituMul {
 public:
  explicit SituMul(const LayerCreationContext& creation_context);
  ~SituMul();

  Status Forward(Tensor fused_tensor, std::vector<Tensor>& output_tensors, const LayerExecContext& exec_ctx = {});

 protected:
  std::shared_ptr<BaseLayer> situ_mul_layer_;
};

}  // namespace ksana_llm
