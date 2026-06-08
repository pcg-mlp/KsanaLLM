/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "layers/matmul_base_layer.h"
#include "models/base/layer_creation_context.h"

namespace ksana_llm {

class Bmm {
 public:
  // Disable a default constructor
  Bmm(const std::string& weight_name, const LayerCreationContext& creation_context,
      const LinearComputeBackend& linear_compute_backend);

  ~Bmm() = default;

  Status Forward(const Tensor& input, Tensor& output);

 protected:
  std::shared_ptr<MatMulBaseLayer> bmm_layer_;
  Tensor weight_;

  int rank_;
  std::shared_ptr<Context> context_;
};
}  // namespace ksana_llm
