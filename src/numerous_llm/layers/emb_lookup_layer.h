/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#include "numerous_llm/layers/base_layer.h"
namespace numerous_llm {

class EmbLookupLayer : public BaseLayer {
 public:
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;
};

}  // namespace numerous_llm
