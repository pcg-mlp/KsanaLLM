/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/layers/base_layer.h"

namespace numerous_llm {

class RotaryEmbeddingLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 protected:
  int max_position_embeddings_;
  int cos_sin_cache_block_id_;
  // RotaryEmbeddingCuda<half> rotary_embedding_cuda_;
};

}  // namespace numerous_llm
