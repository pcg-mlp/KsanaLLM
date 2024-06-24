/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/attention_layer.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
class FlashAttentionLayer : public AttentionLayer<SCALAR_T> {
 public:
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 protected:
  using AttentionLayer<SCALAR_T>::rank_;
  using AttentionLayer<SCALAR_T>::block_token_num_;
  using AttentionLayer<SCALAR_T>::context_;
  using AttentionLayer<SCALAR_T>::num_heads_;
  using AttentionLayer<SCALAR_T>::stride_size_;
  using AttentionLayer<SCALAR_T>::tensor_para_size_;
  using AttentionLayer<SCALAR_T>::kv_cache_dtype_;
  using AttentionLayer<SCALAR_T>::is_causal_;
  using AttentionLayer<SCALAR_T>::layer_index_;
  using AttentionLayer<SCALAR_T>::layer_num_;
  using AttentionLayer<SCALAR_T>::block_size_;
  using AttentionLayer<SCALAR_T>::num_kv_heads_;
  using AttentionLayer<SCALAR_T>::head_size_;
#ifdef ENABLE_CUDA
  using AttentionLayer<SCALAR_T>::rotary_embedding_cuda_;
  using AttentionLayer<SCALAR_T>::alibi_slopes_;
#endif

#ifdef ENABLE_ACL
  using AttentionLayer<SCALAR_T>::workspace_block_id_;
  using AttentionLayer<SCALAR_T>::workspace_size_;
#endif
};

}  // namespace ksana_llm
