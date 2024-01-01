/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/layers/base_layer.h"

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"

namespace numerous_llm {

class RotaryEmbeddingLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 protected:
  int max_position_embeddings_;
  int cos_sin_cache_block_id_;
  llm_kernels::nvidia::RotaryEmbeddingCuda<half> rotary_embedding_cuda_;
  half* cos_sin_cache_ptr_;
};

}  // namespace numerous_llm
