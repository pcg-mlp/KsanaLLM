/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/attention_layer.h"

namespace ksana_llm {

struct PagedMlaAttentionInputs {
  const Tensor* hidden_buffer_1;
  const Tensor* kv_seq_len;
  const Tensor* kv_list;
  const Tensor* cache_offset;
  const Tensor* rotary_embedding_pos;
  const Tensor* rotary_embedding_mask;
  const Tensor* layer_kv_cache;
  const Tensor* block_table;
  const Tensor* q_nope;
  const Tensor* q_pe;
  const Tensor* compressed_kv;
  const Tensor* k_pe;
  const Tensor* tile_scheduler_metadata;
  const Tensor* num_splits;
#ifdef ENABLE_KUNLUN
  // Kunlun-specific fields (not used by CUDA/Ascend)
  const Tensor* seq_len_host;        // context lengths, HOST
  const Tensor* k_cache_blocks_base;  // kv_cache base (NO layer offset), DEVICE
  const Tensor* layers_slot_mapping;  // per-token slot mapping, DEVICE
  const Tensor* layers_block_table;   // per-layer block table, DEVICE
  const Tensor* atb_attention_attr;   // [slot_mapping_dim_1, max_blocks_per_query]
  const Tensor* context_len_device;  // DEVICE: context lengths [batch_size], int32
#endif
};

class PagedMlaAttentionLayer : public AttentionLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  Status Forward(const PagedMlaAttentionInputs& inputs, Tensor* output);

 private:
  template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
  Status ForwardT(const PagedMlaAttentionInputs& inputs, Tensor* output);
};

}  // namespace ksana_llm
