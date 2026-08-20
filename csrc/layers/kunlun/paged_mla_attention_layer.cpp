/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/paged_mla_attention_layer.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "device/device_utils.h"
#include "device/kunlun/kunlun_device_context_impl.h"
#include "kernels/kunlun/attention_kernel_wrapper.h"
#include "layers/attention_layer.h"
#include "utils/logger.h"

namespace xpukernel = xpytorch::xpu::api;

namespace ksana_llm {

Status PagedMlaAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                    std::shared_ptr<Context> context, int rank) {
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}

Status PagedMlaAttentionLayer::Forward(const PagedMlaAttentionInputs& inputs, Tensor* output) {
  DISPATCH_BY_DTYPE_AND_KVTYPE(inter_data_type_, kv_cache_dtype_, ForwardT, inputs, output);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status PagedMlaAttentionLayer::ForwardT(const PagedMlaAttentionInputs& inputs, Tensor* output) {
  using T = SCALAR_T;

  // Parse inputs — Kunlun protocol (differs from NVIDIA):
  // Kunlun passes HOST seq_len, per-token slot_mapping, per-layer block_table, and atb_attention_attr.
  const Tensor& hidden_buffer_1 = *inputs.hidden_buffer_1;  // [0] workspace
  const Tensor& seq_len_host = *inputs.seq_len_host;        // [1] HOST: context lengths [batch_size], int32
  const Tensor& k_cache_blocks_base =
      *inputs.k_cache_blocks_base;  // [2] kv_cache global base (NO layer offset), DEVICE
  const Tensor& layers_slot_mapping =
      *inputs.layers_slot_mapping;  // [3] per-token slot mapping, DEVICE [layer_num * dim_1]
  const Tensor& rotary_embedding_pos = *inputs.rotary_embedding_pos;
  const Tensor& rotary_embedding_mask = *inputs.rotary_embedding_mask;
  const Tensor& layer_kv_cache = *inputs.layer_kv_cache;  // [6] layer_kv_cache metadata (block_num + per-layer ptrs)
  const Tensor& layers_block_table =
      *inputs.layers_block_table;                // [7] per-layer block table, DEVICE (values include layer offset)
  const Tensor& q_nope_tensor = *inputs.q_nope;  // [8] [batch*q_seq_len, num_heads, kv_lora_rank]
  const Tensor& q_pe_tensor = *inputs.q_pe;      // [9] [batch*q_seq_len, num_heads, qk_rope_head_dim]
  const Tensor& compressed_kv_tensor = *inputs.compressed_kv;               // [10] [batch*q_seq_len, kv_lora_rank]
  const Tensor& k_pe_tensor = *inputs.k_pe;                                 // [11] [batch*q_seq_len, qk_rope_head_dim]
  const Tensor& tile_scheduler_metadata = *inputs.tile_scheduler_metadata;  // [12] NVIDIA-specific, Kunlun ignores
  const Tensor& num_splits = *inputs.num_splits;                            // [13] NVIDIA-specific, Kunlun ignores
  const Tensor& atb_attention_attr =
      *inputs.atb_attention_attr;  // [14] HOST: [slot_mapping_dim_1, max_num_blocks_per_query]
  const Tensor& context_len_device = *inputs.context_len_device;  // [15] DEVICE: context lengths [batch_size], int32

  (void)tile_scheduler_metadata;
  (void)num_splits;

  // In decode phase, each sequence produces q_seq_len tokens (typically 1).
  // total_tokens = batch_size * q_seq_len.
  // NOTE: seq_len_host.shape[0] is max_batch_size (pre-allocated), NOT actual batch_size.
  // Use k_pe_tensor.shape[0] as total_tokens; for decode q_seq_len=1 so batch_size=total_tokens.
  const size_t total_tokens = k_pe_tensor.shape[0];
  const size_t q_seq_len = 1;  // decode path: always 1 token per sequence
  const size_t batch_size = total_tokens / q_seq_len;

  xdnn::Context* xctx = static_cast<xdnn::Context*>(
      context_->device->ext->GetXdnnContext(context_->device->GetPrimaryComputeStream().Get()));
  auto* xpuctx = static_cast<xpukernel::Context*>(
      context_->device->ext->GetXpuKernelContext(context_->device->GetPrimaryComputeStream().Get()));

  // Workspace layout in hidden_buffer_1:
  // [q_concat: total_tokens * num_heads * (kv_lora_rank + qk_rope_head_dim)]
  // [k_concat: total_tokens * 1 * (kv_lora_rank + qk_rope_head_dim)]
  const int64_t q_head_dim = kv_lora_rank_ + qk_rope_head_dim_;
  const int64_t k_head_dim = kv_lora_rank_ + qk_rope_head_dim_;

  // Compute softmax scale: attn_scale_ is pre-computed in Init (DeepSeek yarn mscale).
  float softmax_scale = attn_scale_;
  T* workspace = hidden_buffer_1.GetPtr<T>();
  T* q_workspace = workspace;
  T* k_workspace = workspace + total_tokens * num_heads_ * q_head_dim;

  const T* q_nope_ptr = q_nope_tensor.GetPtr<T>();
  const T* q_pe_ptr = q_pe_tensor.GetPtr<T>();
  const T* compressed_kv_ptr = compressed_kv_tensor.GetPtr<T>();
  const T* k_pe_ptr = k_pe_tensor.GetPtr<T>();
  int64_t* positions = rotary_embedding_pos.GetPtr<int64_t>();

  // Step 1: Concat q = cat(q_nope_projected, q_pe)
  // q_nope: [total_tokens * num_heads, kv_lora_rank]
  // q_pe: [total_tokens * num_heads, qk_rope_head_dim]
  // → q_workspace: [total_tokens * num_heads, kv_lora_rank + qk_rope_head_dim]
  int ret;
  {
    const int64_t outer = static_cast<int64_t>(total_tokens * num_heads_);
    const int64_t c1 = static_cast<int64_t>(kv_lora_rank_);
    const int64_t c2 = static_cast<int64_t>(qk_rope_head_dim_);
    std::vector<const T*> x_list = {q_nope_ptr, q_pe_ptr};
    std::vector<std::vector<int64_t>> xshape_list = {{outer, c1}, {outer, c2}};
    ret = XpuConcat<T>(xpuctx, x_list, q_workspace, xshape_list, /*axis=*/1);
  }
  if (ret != 0) {
    KLLM_LOG_ERROR << "PagedMlaAttentionLayer: XpuConcat for q failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "XpuConcat q failed");
  }

  // Step 2: Prepare k = [total_tokens, kv_lora_rank + qk_rope_head_dim]
  if (k_pe_tensor.shape[1] == kv_lora_rank_ + qk_rope_head_dim_) {
    // rmsnorm_split_col path: k_pe_tensor is already the full latent_cache [tokens, kv_lora+rope]
    // Safe to use in-place: prefill already consumed the buffer, decode pages don't overlap.
    k_workspace = const_cast<T*>(k_pe_tensor.GetPtr<T>());
  } else {
    // Original path: concat compressed_kv and k_pe
    // compressed_kv: [total_tokens, kv_lora_rank]
    // k_pe: [total_tokens, qk_rope_head_dim]
    // → k_workspace: [total_tokens, kv_lora_rank + qk_rope_head_dim]
    const int64_t outer = static_cast<int64_t>(total_tokens);
    const int64_t c1 = static_cast<int64_t>(kv_lora_rank_);
    const int64_t c2 = static_cast<int64_t>(qk_rope_head_dim_);
    std::vector<const T*> x_list = {compressed_kv_ptr, k_pe_ptr};
    std::vector<std::vector<int64_t>> xshape_list = {{outer, c1}, {outer, c2}};
    ret = XpuConcat<T>(xpuctx, x_list, k_workspace, xshape_list, /*axis=*/1);
    if (ret != 0) {
      KLLM_LOG_ERROR << "PagedMlaAttentionLayer: XpuConcat for k failed, ret=" << ret;
      return Status(RET_RUNTIME_FAILED, "XpuConcat k failed");
    }
  }

  // Step 3: dpsk_decode_rotary_embedding (in-place on q and k's last rope dims)
  const float* cos_sin_cache = reinterpret_cast<const float*>(cos_sin_cache_ptr_);
  // query: [total_tokens, num_heads, kv_lora_rank + qk_rope_head_dim]
  //   stride = kv_lora_rank + qk_rope_head_dim, offset = kv_lora_rank
  // key: [total_tokens, 1, kv_lora_rank + qk_rope_head_dim]
  //   stride = kv_lora_rank + qk_rope_head_dim, offset = kv_lora_rank
  int64_t max_seq_len = 0;  // Will be derived from cos_sin_cache shape; use a large value
  // The cos_sin_cache is [max_position_embeddings, rot_dim] stored during init
  max_seq_len = static_cast<int64_t>(max_position_embeddings_);

  ret = XpuDpskDecodeRotaryEmbedding<T, float, int64_t>(
      xctx, q_workspace, k_workspace, cos_sin_cache, positions, max_seq_len,
      static_cast<int64_t>(qk_rope_head_dim_),  // rot_dim
      static_cast<int64_t>(q_head_dim),         // query_stride
      static_cast<int64_t>(k_head_dim),         // key_stride
      static_cast<int64_t>(kv_lora_rank_),      // query_offset (rope starts after nope)
      static_cast<int64_t>(kv_lora_rank_),      // key_offset
      static_cast<int64_t>(num_heads_),
      1,  // num_kv_heads = 1 (compressed cache has single "head")
      static_cast<int64_t>(total_tokens));
  if (ret != 0) {
    KLLM_LOG_ERROR << "PagedMlaAttentionLayer: dpsk_decode_rotary_embedding failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "dpsk_decode_rotary_embedding failed");
  }

  // Step 4: KV cache write using per-token slot_mapping from layers_slot_mapping
  // slot_mapping values are absolute slot offsets (include layer offset),
  // so kv_cache base must be the global base (NO layer offset).
  uint64_t slot_mapping_dim_1 = atb_attention_attr.GetPtr<uint64_t>()[0];
  int32_t* slot_mapping = layers_slot_mapping.GetPtr<int32_t>() + this->layer_index_ * slot_mapping_dim_1;
  const int64_t kv_cache_block_num = *(layer_kv_cache.GetPtr<int64_t>());
  T* kv_cache_base = k_cache_blocks_base.GetPtr<T>();
  int64_t kv_cache_dim = kv_lora_rank_ + qk_rope_head_dim_;

  ret = XpuMlaReshapeCachedKV<T>(xctx, k_workspace, kv_cache_base, slot_mapping, static_cast<int64_t>(total_tokens),
                                 kv_cache_block_num, kv_cache_dim, static_cast<int64_t>(block_token_num_));
  if (ret != 0) {
    KLLM_LOG_ERROR << "PagedMlaAttentionLayer: XpuMlaReshapeCachedKV failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "XpuMlaReshapeCachedKV failed");
  }

  // Step 5: MLA decode attention (XpuPagedMlaAttention)
  // Compute max_context_len from HOST data (safe CPU access — seq_len_host is HOST tensor)
  int32_t* context_len_host = seq_len_host.GetPtr<int32_t>();
  int64_t max_context_len = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    max_context_len = std::max(max_context_len, static_cast<int64_t>(context_len_host[i]));
  }

  // Block table: per-layer offset from layers_block_table
  // Values in block_table already contain layer offset, so kv_cache must be global base.
  uint64_t max_num_blocks_per_query = atb_attention_attr.GetPtr<uint64_t>()[1];
  const int32_t* block_table_ptr =
      layers_block_table.GetPtr<int32_t>() + this->layer_index_ * batch_size * max_num_blocks_per_query;
  const int max_blocks_per_seq = static_cast<int>(max_num_blocks_per_query);

  // Debug: print parameters for layer 0 only
  if (this->layer_index_ == 0) {
    KLLM_LOG_INFO << "PagedMlaAttention DEBUG layer=0"
                  << " batch_size=" << batch_size << " q_seq_len=" << q_seq_len << " num_heads=" << num_heads_
                  << " kv_lora_rank=" << kv_lora_rank_ << " qk_rope_head_dim=" << qk_rope_head_dim_
                  << " block_token_num=" << block_token_num_ << " max_context_len=" << max_context_len
                  << " max_blocks_per_seq=" << max_blocks_per_seq << " kv_cache_block_num=" << kv_cache_block_num
                  << " softmax_scale=" << softmax_scale << " context_len_host[0]=" << context_len_host[0];
  }

  ret = XpuPagedMlaAttention<T, int32_t>(
      xctx, output->GetPtr<T>(), q_workspace, kv_cache_base, block_table_ptr, context_len_device.GetPtr<int32_t>(),
      context_len_host, static_cast<int64_t>(batch_size), static_cast<int64_t>(q_seq_len),
      static_cast<int64_t>(num_heads_), static_cast<int64_t>(kv_lora_rank_), static_cast<int64_t>(qk_rope_head_dim_),
      softmax_scale, static_cast<int64_t>(block_token_num_), max_context_len, static_cast<int64_t>(max_blocks_per_seq));
  if (ret != 0) {
    KLLM_LOG_ERROR << "PagedMlaAttentionLayer: XpuPagedMlaAttention failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "XpuPagedMlaAttention failed");
  }

  // Output shape: [total_tokens, num_heads, kv_lora_rank]
  output->shape = {total_tokens, static_cast<size_t>(num_heads_), static_cast<size_t>(kv_lora_rank_)};
  return Status();
}

// Explicit instantiations for fp16 and bf16. KV_DTYPE=kAuto is the only path on Kunlun.
template Status PagedMlaAttentionLayer::ForwardT<float16, float16, llm_kernels::utils::KVCacheType::kAuto>(
    const PagedMlaAttentionInputs&, Tensor*);
template Status PagedMlaAttentionLayer::ForwardT<bfloat16, bfloat16, llm_kernels::utils::KVCacheType::kAuto>(
    const PagedMlaAttentionInputs&, Tensor*);

}  // namespace ksana_llm
