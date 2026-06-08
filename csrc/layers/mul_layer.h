/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

// Pointwise multiplication of two tensors
class MulLayer : public BaseLayer {
 public:
  Status Forward(const Tensor* a, const Tensor* b, Tensor* output, const LayerExecContext& exec_ctx = {});

 private:
  template <typename T>
  Status ForwardT(const Tensor& a, const Tensor& b, Tensor& output);
};

}  // namespace ksana_llm
