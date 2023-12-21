/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/layers/attention_layer.h"

namespace numerous_llm {

class FlashAttentionLayer : public AttentionLayer {
 public:
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;
};

}  // namespace numerous_llm
