/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_CUDA
#  include <optional>

#  include "csrc/kernels/nvidia/alibi/alibi.h"
#  include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#endif

#ifdef ENABLE_ACL
#  include "csrc/kernels/ascend/paged_attention/paged_attention.h"
#endif

#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

template <typename T, template <typename, typename, bool> class ATTENTION_LAYER>
std::shared_ptr<BaseLayer> CreateAttentionLayer(DataType kv_cache_dtype) {
  switch (kv_cache_dtype) {
    case TYPE_FP8_E5M2:
      return std::make_shared<ATTENTION_LAYER<T, uint8_t, true>>();
    default:
      return std::make_shared<ATTENTION_LAYER<T, T, false>>();
  }
}

template <typename T>
class AttentionLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) = 0;

#ifdef ENABLE_ACL

 public:
  // The attention implementation for ascend device.
  static std::shared_ptr<llm_kernels::ascend::PagedAttention<T>> ascend_paged_attn_;
#endif

 protected:
  int layer_num_;
  int layer_index_;
  int block_size_;
  int block_token_num_;
  int num_heads_;
  int num_kv_heads_;
  int head_size_;
  int stride_size_;
  int tensor_para_size_;

  // kv_cache storage type
  DataType kv_cache_dtype_;

  bool is_causal_{true};
#ifdef ENABLE_CUDA
  llm_kernels::nvidia::RotaryEmbeddingCuda<T> rotary_embedding_cuda_;
  std::optional<void*> alibi_slopes_ = {};
#endif

#ifdef ENABLE_ACL
  // NOTE(karlluo): only need by ascend
  int workspace_block_id_{-1};
  size_t workspace_size_{0ul};

  void PrepareWorkspaceBuffer(const size_t workspace_needed, void* workspace_buf_ptr);
#endif
};

}  // namespace ksana_llm
