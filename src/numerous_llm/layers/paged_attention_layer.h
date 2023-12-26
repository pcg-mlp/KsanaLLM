/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/layers/attention_layer.h"
#include "numerous_llm/layers/rotary_embedding_layer.h"

namespace numerous_llm {

class PagedAttentionLayer : public AttentionLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 protected:
  RotaryEmbeddingLayer rotary_embedding_layer_;
};

}  // namespace numerous_llm
