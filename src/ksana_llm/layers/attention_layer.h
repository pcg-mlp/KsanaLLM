/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_CUDA
#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#endif
#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

class AttentionLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) = 0;

 protected:
  int layer_index_;
  int block_size_;
  int block_token_num_;
  int num_heads_;
  int num_kv_heads_;
  int head_size_;
  int stride_size_;
  bool is_causal_{true};
  #ifdef ENABLE_CUDA
  llm_kernels::nvidia::RotaryEmbeddingCuda<half> rotary_embedding_cuda_;
  #endif
};

}  // namespace ksana_llm
