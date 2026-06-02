/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

template <typename T>
class PermuteLayer : public BaseLayer {
 public:
  std::vector<size_t> FindPermutation(const std::vector<size_t>& input_shape, const std::vector<size_t>& output_shape);

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx = {}) override;
};

}  // namespace ksana_llm