/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

template <typename T>
class PermuteLayer : public BaseLayer {
 public:
  std::vector<size_t> FindPermutation(const std::vector<size_t>& input_shape, const std::vector<size_t>& output_shape);

  Status Forward(const Tensor* input, Tensor* output, const LayerExecContext& exec_ctx = {});
};

}  // namespace ksana_llm
