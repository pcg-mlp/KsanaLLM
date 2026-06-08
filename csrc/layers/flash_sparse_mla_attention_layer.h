/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/attention_layer.h"

namespace ksana_llm {

struct FlashSparseMlaAttentionInputs {
  const Tensor* rotary_embedding_pos;
  const Tensor* rotary_embedding_mask;
  const Tensor* q_pe;
  const Tensor* k_pe;
  const Tensor* q_nope;
  const Tensor* q;
  const Tensor* compressed_kv;
  const Tensor* kv_list;
  const Tensor* prefix_offsets;
  const Tensor* seqlens_without_prefix;
  const Tensor* cache_offset;
  const Tensor* layer_kv_cache;
  const Tensor* block_table;
  const Tensor* tile_scheduler_metadata;
  const Tensor* num_splits;
  const Tensor* indices;
};

class FlashSparseMlaAttentionLayer : public AttentionLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  Status Forward(const FlashSparseMlaAttentionInputs& inputs, Tensor* output);

  virtual size_t GetWorkspaceSize() override;

 private:
  template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
  Status ForwardT(const FlashSparseMlaAttentionInputs& inputs, Tensor* output);

  size_t max_token_num_ = 0;
};

}  // namespace ksana_llm
