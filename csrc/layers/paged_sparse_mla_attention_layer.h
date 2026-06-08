/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/attention_layer.h"

namespace ksana_llm {

struct PagedSparseMlaAttentionInputs {
  const Tensor* rotary_embedding_pos;
  const Tensor* rotary_embedding_mask;
  const Tensor* q_pe;
  const Tensor* k_pe;
  const Tensor* q_nope;
  const Tensor* q;
  const Tensor* compressed_kv;
  const Tensor* kv_list;
  const Tensor* kv_seq_len;
  const Tensor* cache_offset;
  const Tensor* layer_kv_cache;
  const Tensor* block_table;
  const Tensor* tile_scheduler_metadata;
  const Tensor* num_splits;
  const Tensor* indices;
};

class PagedSparseMlaAttentionLayer : public AttentionLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  Status Forward(const PagedSparseMlaAttentionInputs& inputs, Tensor* output);

 private:
  template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
  Status ForwardT(const PagedSparseMlaAttentionInputs& inputs, Tensor* output);
};

}  // namespace ksana_llm
