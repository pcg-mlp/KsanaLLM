/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/attention_layer.h"

namespace ksana_llm {

class FlashAttentionLayer : public AttentionLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;
#ifdef ENABLE_CUDA
  // Typed parameters for the CUDA forward path.
  struct CudaForwardParams {
    const Tensor* qkv;
    const Tensor* token_offset;
    const Tensor* kv_list;
    const Tensor* prefix_offset;
    const Tensor* kv_cache_offset;
    const Tensor* rotary_embedding_pos;
    const Tensor* rotary_embedding_mask;
    const Tensor* src_flexible_rotary_embedding_pos;
    const Tensor* flexible_rotary_embedding_mask;
    const Tensor* dst_flexible_kv_cache;
    const Tensor* src_flexible_kv_cache;
    const Tensor* dst_flexible_token_idx;
    const Tensor* src_flexible_token_idx;
    const Tensor* flexible_offset_uint64;
    // Scalars formerly encoded in forward_shape (input[14]):
    int layer_block_num;  // shape[2]
    // Number of decode-stage TOKENS in this batch (shape[12], dp_decode_tokens). Used to carve the
    // prefill region out of the shared qkv buffer: prefill_tokens = qkv->shape[0] - decode_token_num.
    // This must be a token count, not a request count: with MTP verify each decode request carries
    // 1 + draft_num tokens, so the request count (shape[3]) over-counts the prefill region in any
    // mixed prefill+decode batch and AttenVarlen's per-token kernels then run past its end.
    int decode_token_num;
    int max_forwarding_tokens;  // shape[6]
    int batch_size;             // shape[7]
    int max_tokens;             // shape[8]
    const Tensor* query_layernorm_weight;
    const Tensor* key_layernorm_weight;
    bool use_cache;
    // Optional blocked-prefill / flashinfer-decode path (nullptr = disabled):
    const Tensor* kv_cache_base_ptr;
    const Tensor* block_table;
    const Tensor* input_without_prefix_offset;
  };
  Status Forward(const CudaForwardParams& params, Tensor* output, const LayerExecContext& exec_ctx = {});
#endif

#ifdef ENABLE_ACL
  // Typed parameters for the Ascend ATB forward path.
  struct AclForwardParams {
    const Tensor* qkv;
    const Tensor* rotary_embedding_pos;
    const Tensor* layers_slot_mapping;
    const Tensor* k_cache_blocks_base;
    const Tensor* v_cache_blocks_base;
    const Tensor* seq_len_host;
    // Scalars formerly encoded in forward_shape (input[6]):
    int batch_size;       // shape[0]
    int max_tokens;       // shape[1]
    int layer_block_num;  // shape[2]
    const Tensor* atb_attention_attr;
  };
  Status Forward(const AclForwardParams& params, Tensor* output, const LayerExecContext& exec_ctx = {});
#endif

#ifdef ENABLE_KUNLUN
  // Typed parameters for the Kunlun XPU forward path.
  // Same tensor protocol as Ascend/ATB, plus QK Norm weights.
  struct KunlunForwardParams {
    const Tensor* qkv;
    const Tensor* rotary_embedding_pos;
    const Tensor* layers_slot_mapping;
    const Tensor* k_cache_blocks_base;
    const Tensor* v_cache_blocks_base;
    const Tensor* seq_len_host;
    // Scalars formerly encoded in forward_shape (input[6]):
    int batch_size;       // shape[0]
    int max_tokens;       // shape[1]
    int layer_block_num;  // shape[2]
    const Tensor* atb_attention_attr;
    // QK Norm weights (Kunlun-specific, for use_qk_norm)
    const Tensor* query_layernorm_weight;
    const Tensor* key_layernorm_weight;
    // HOST int32 cumulative q offsets [batch_size+1] (avoids per-layer D2H cumsum)
    const Tensor* qlod;
  };
  Status Forward(const KunlunForwardParams& params, Tensor* output, const LayerExecContext& exec_ctx = {});
#endif

 private:
#ifdef ENABLE_CUDA
  template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
  Status ForwardT(const CudaForwardParams& p, Tensor* output);
#endif
#ifdef ENABLE_KUNLUN
  template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
  Status ForwardT(const KunlunForwardParams& p, Tensor* output);
#endif

 private:
#ifdef ENABLE_ACL
  // ATB attention implement for ascend device, for ATB run with graph, each layer need's one kernel implement
  // instance
  std::shared_ptr<llm_kernels::ascend::ATBAttention> atb_flash_attn_;
#endif
  bool enable_blocked_multi_token_forwarding_kv_;
  bool use_flashinfer_for_decode_;
  bool use_hpc_attention_ = false;
};

}  // namespace ksana_llm
