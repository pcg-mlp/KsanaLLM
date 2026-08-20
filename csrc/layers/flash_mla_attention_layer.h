/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/attention_layer.h"

namespace ksana_llm {

struct FlashMlaAttentionInputs {
  const Tensor* dp_input_offset;
  const Tensor* dp_input_offset_host;
  const Tensor* kv_list;
  const Tensor* dp_input_prefix;
  const Tensor* dp_prefill_q_offset;
  const Tensor* dp_prefill_q_offset_host;
  const Tensor* chunk_kv_batch_starts_host;
  const Tensor* chunk_kv_cu_extend_lens;
  const Tensor* chunk_kv_cu_seq_lens;
  const Tensor* kv_cache_offset;
  const Tensor* rotary_embedding_pos;
  const Tensor* rotary_embedding_mask;
  const Tensor* dp_src_flexible_rotary_embedding_pos;
  const Tensor* dp_dst_flexible_rotary_embedding_pos;
  const Tensor* dp_flexible_rotary_embedding_mask;
  const Tensor* dp_dst_flexible_kv_cache;
  const Tensor* dp_src_flexible_kv_cache;
  const Tensor* dp_dst_flexible_token_idx;
  const Tensor* dp_src_flexible_token_idx;
  const Tensor* dp_flexible_offset_uint64;
  const Tensor* forward_shape;
  const Tensor* q_nope_rope;
  const Tensor* kv_buffer;
  const Tensor* k_rope_buffer;
  const Tensor* kv_b_nope_proj_weight;
  const Tensor* v_head_proj_weight;
  const Tensor* k_buffer;
  const Tensor* v_buffer;
#ifdef ENABLE_KUNLUN
  // Kunlun-specific fields (not used by CUDA/Ascend)
  const Tensor* dp_input_offset_int32;       // device int32 offsets (for xfa VectorParam ATTN_LVSL)
  const Tensor* dp_prefill_q_offset_int32;   // device int32 offsets (for xfa VectorParam ATTN_LVSL)
  const Tensor* layer_kv_cache;              // layer kv cache ptr (for block_num metadata)
  const Tensor* block_table;                 // block table (unused on Kunlun, kept for interface compat)
  const Tensor* prefix_kv_buffer;            // prefix kv buffer (unused on Kunlun, no prefix cache)
  const Tensor* qlod_host_tensor;            // HOST int32 cumulative q offsets [batch_size+1]
  const Tensor* kvlod_host_tensor;           // HOST int32 cumulative kv offsets [batch_size+1]
  const Tensor* layers_slot_mapping;         // per-layer per-token slot mapping (device)
  const Tensor* atb_attention_attr;          // [slot_mapping_dim_1, max_num_blocks_per_query]
  const Tensor* prefix_kvlod_host_tensor;     // HOST int32 prefix cumulative lengths [batch_size+1]
  const Tensor* prefix_kvlod_dev_tensor;     // DEVICE int32 prefix cumulative lengths [batch_size+1]
  const Tensor* kv_b_proj_combined_weight;   // weight for prefix kv_b_proj
#endif
};

class FlashMlaAttentionLayer : public AttentionLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;
  Status Forward(const FlashMlaAttentionInputs& inputs, Tensor* output);

  virtual size_t GetWorkspaceSize() override;

 private:
  template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
  Status ForwardT(const FlashMlaAttentionInputs& inputs, Tensor* output);

  // NOTE: the per-chunk prefix-cache helper (previously ForwardChunkT) lives as a file-local
  // free function in kunlun/flash_mla_attention_layer.cpp. It is deliberately NOT declared here
  // so that non-Kunlun translation units including this header never reference xdnn::Context.

 private:
  size_t max_step_token_num_ = 0;
  size_t max_chunked_kv_tokens_ = 0;
};

}  // namespace ksana_llm
