/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#include "numerous_llm/layers/base_layer.h"

namespace numerous_llm {

class AttentionLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) = 0;

 protected:
  int layer_index_;
  int max_position_embeddings_;
  int block_size_;
  int block_token_num_;
  int num_heads_;
  int num_kv_heads_;
  int head_size_;
  bool is_causal_{true};
  int cos_sin_cache_block_id_;
  llm_kernels::nvidia::RotaryEmbeddingCuda<half> rotary_embedding_cuda_;
  half* cos_sin_cache_ptr_;
};

}  // namespace numerous_llm
