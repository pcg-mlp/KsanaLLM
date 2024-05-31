/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <torch/torch.h>
#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

template <typename T>
class SubinputLayer : public BaseLayer {
 public:
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;
  void Clear() { cast_tensor_vec_.clear(); }

 public:
  std::vector<torch::Tensor> cast_tensor_vec_;
};

}  // namespace ksana_llm
