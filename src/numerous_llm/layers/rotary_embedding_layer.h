/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/layers/base_layer.h"

namespace numerous_llm {

class RotaryEmbeddingLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, cudaStream_t stream) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 protected:
  int max_position_embeddings_;
};

}  // namespace numerous_llm
