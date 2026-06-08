/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

// Common interface for all matmul-style layers that operate on a single (input, weight) pair.
// Callers (e.g. Linear, Bmm) hold a MatMulBaseLayer* so they can use the pointer-based
// Forward without constructing temporary std::vector<Tensor>.
class MatMulBaseLayer : public BaseLayer {
 public:
  virtual Status Forward(const Tensor* input, const Tensor* weight, Tensor* output,
                         const LayerExecContext& exec_ctx = {}) = 0;
};

}  // namespace ksana_llm
