/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/attention_layer.h"

namespace ksana_llm {

template <typename T>
class FlashAttentionLayer : public AttentionLayer<T> {
 public:
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 protected:
  using AttentionLayer<T>::rank_;
  using AttentionLayer<T>::block_token_num_;
  using AttentionLayer<T>::context_;
  using AttentionLayer<T>::num_heads_;
  using AttentionLayer<T>::stride_size_;
  using AttentionLayer<T>::tensor_para_size_;
  using AttentionLayer<T>::is_causal_;
  using AttentionLayer<T>::layer_index_;
  using AttentionLayer<T>::block_size_;
  using AttentionLayer<T>::num_kv_heads_;
  using AttentionLayer<T>::head_size_;
#ifdef ENABLE_CUDA
  using AttentionLayer<T>::rotary_embedding_cuda_;
  using AttentionLayer<T>::alibi_slopes_;
#endif

#ifdef ENABLE_ACL
  using AttentionLayer<T>::workspace_block_id_;
  using AttentionLayer<T>::workspace_size_;
#endif
};

}  // namespace ksana_llm
