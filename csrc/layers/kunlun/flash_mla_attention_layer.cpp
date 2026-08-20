/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/flash_mla_attention_layer.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "device/device_utils.h"
#include "device/kunlun/kunlun_device_context_impl.h"
#include "kernels/kunlun/attention_kernel_wrapper.h"
#include "kernels/kunlun/quant_gemm_wrapper.h"
#include "kernels/permute.h"
#include "layers/attention_layer.h"
#include "utils/logger.h"
#include "utils/tensor.h"

namespace xpukernel = xpytorch::xpu::api;

namespace ksana_llm {

// ════════════════════════════════════════════════════════════════════════════
// Buffer Layout Overview (prefix cache path)
// ════════════════════════════════════════════════════════════════════════════
//
// Independent buffers (never conflict):
//   q_ptr          = q_nope_rope_tensor    (mla_buffers.kv_lora_or_q_nope_rope_buffer)
//   latent_cache   = k_rope_buffer         (mla_buffers.k_rope_buffer, also LSE storage)
//   kv_compressed  = kv_buffer_tensor      (mla_buffers.kv_buffer, Step 0 input only)
//
// Swappable buffers (the two large hidden buffers + reduce buffer):
//   hidden_buffer_0 = output tensor        (from multihead_latent_attention.cpp attn_output_tensors)
//   hidden_buffer_1 = k_buffer             (from multihead_latent_attention.cpp hidden_buffer_tensors_1)
//   reduce_buffer   = v_buffer             (from multihead_latent_attention.cpp reduce_buffer_tensors / shared_buffer)
//
// Step 0 (flash_mla_attention.cpp, before layer):
//   kv_b_proj_combined_(kv_compressed → hidden_buffer_1 if prefix, else hidden_buffer_0)
//
// Step 1 (ForwardT, rope):
//   Input:  kv (hidden_buffer_1 if prefix, else hidden_buffer_0), q_ptr, latent_cache (independent)
//   Output: new_k → hidden_buffer_0 if prefix, else hidden_buffer_1
//           new_v → reduce_buffer (always)
//   Constraint: kv input ≠ k/v output ✓
//
// Step 2 (ForwardT, cache write):
//   Input:  latent_cache (independent, roped)
//   Output: paged cache blocks (independent)
//
// ── No prefix path (Step 2.5) ──
//   Attention: q_ptr × (new_k in hidden_buffer_1, new_v in reduce_buffer) → output (hidden_buffer_0)
//   Constraint: k/v input (hidden_buffer_1/reduce_buffer) ≠ output (hidden_buffer_0) ✓
//
// ── Prefix path (ForwardChunkT, per chunk) ──
//   Phase 3 (Gather):
//     Input:  paged cache blocks
//     Output: layer_workspace [prefix_kv_c: T_p_ch * kv_lora_rank][prefix_k_pe: T_p_ch * qk_rope]
//
//   Phase 4 (kv_b_proj for prefix):
//     Input:  layer_workspace prefix_kv_c [T_p_ch * kv_lora_rank]
//     Output: layer_workspace prefix_kv_proj [T_p_ch * H * (qk_nope + v_head)]
//     (prefix_kv_c freed after this phase)
//
//   Phase 5 (Split):
//     Input:  layer_workspace prefix_kv_proj [T_p_ch * H * (qk_nope + v_head)]
//     Output: layer_workspace prefix_k [T_p_ch * H * qk_head_dim]  (k_nope part, gap reserved for k_pe)
//             layer_workspace prefix_v [T_p_ch * H * v_head_dim]
//     (prefix_kv_proj freed after split reads complete)
//
//   Phase 6 (k_pe expand):
//     Input:  layer_workspace prefix_k_pe [T_p_ch * qk_rope]  (from Phase 3)
//     Output: fills gap in prefix_k [T_p_ch * H * qk_head_dim] at offset qk_nope
//     (prefix_k_pe freed after this phase)
//
//   Phase 7 (Self-attention with LSE):
//     Input:  q_ptr (independent, chunk-local), new_k (hidden_buffer_0, chunk-local), new_v (reduce_buffer,
//     chunk-local) Output: o_new → hidden_buffer_1 [T_q_ch * H * v_head]  (chunk-local)
//             lse_new → latent_cache [H * T_q_ch] (float, chunk-local offset)
//     Constraint: k/v input (hidden_buffer_0/reduce_buffer) ≠ o_new output (hidden_buffer_1) ✓
//
//   Phase 8 (Cross-attention with LSE):
//     Input:  q_ptr (independent, chunk-local), prefix_k/prefix_v (layer_workspace)
//     Output: o_prefix → hidden_buffer_1 [T_q_ch * H * v_head]  (after o_new)
//             lse_prefix → latent_cache [H * T_q_ch] (float, after lse_new)
//
//   Phase 9 (Permute LSE):
//     Input/Output: latent_cache (in-place region, [H,T]→[T,H] for both lse_new and lse_prefix)
//     Layout (float, chunk-local from lse_new):
//       [0,      H*T_q_ch): lse_new        (input [H, T_q_ch])
//       [H*T_q_ch, 2*H*T_q_ch): lse_prefix (input [H, T_q_ch])
//       [2*H*T_q_ch, 3*H*T_q_ch): lse_new_T    (output [T_q_ch, H])
//       [3*H*T_q_ch, 4*H*T_q_ch): lse_prefix_T (output [T_q_ch, H])
//
//   Phase 10 (Merge):
//     Input:  o_new (hidden_buffer_1, first half), o_prefix (hidden_buffer_1, second half)
//             lse_new_T, lse_prefix_T (latent_cache)
//     Output: final output → hidden_buffer_0 [T_q_ch * H * v_head]  (chunk-local, new_k no longer needed)
//     Constraint: o_new/o_prefix input (hidden_buffer_1) ≠ output (hidden_buffer_0) ✓
// ════════════════════════════════════════════════════════════════════════════

Status FlashMlaAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                    std::shared_ptr<Context> context, int rank) {
  max_step_token_num_ = runtime_config.max_step_token_num;
  max_chunked_kv_tokens_ = runtime_config.max_chunked_kv_tokens;
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}

size_t FlashMlaAttentionLayer::GetWorkspaceSize() {
  // layer_workspace holds prefix-related temporary data (Phase 3-6).
  // Sized by max_chunked_kv_tokens_ (per-chunk prefix tokens), NOT max_step_token_num_.
  //
  // Sequential layout (no compaction of freed regions):
  //   [prefix_kv_c][prefix_k_pe][prefix_kv_proj][prefix_k][prefix_v]
  //    kv_lora_rank  qk_rope    H*(qk_nope+v_head) H*qk_head_dim H*v_head
  //
  // Phase 5 peak (all live): prefix_k_pe + prefix_kv_proj + prefix_k + prefix_v
  //   = qk_rope + H*(qk_nope+v_head + qk_head_dim + v_head)
  // Layout total adds prefix_kv_c (freed after Phase 4 but not compacted).
  size_t elem_size = GetTypeSize(inter_data_type_);
  size_t per_token = (kv_lora_rank_ + qk_rope_head_dim_ +
                      num_heads_ * (qk_nope_head_dim_ + v_head_dim_ +        // prefix_kv_proj
                                    qk_nope_head_dim_ + qk_rope_head_dim_ +  // prefix_k
                                    v_head_dim_)) *                          // prefix_v
                     elem_size;
  return max_chunked_kv_tokens_ * per_token;
}

namespace {
// Per-chunk prefix-cache helper. Kept as a file-local free function (rather than a member) so the
// shared flash_mla_attention_layer.h never references xdnn::Context in non-Kunlun builds. Layer
// state that the body needs is passed in explicitly (parameter names mirror the member names so the
// body reads identically to the original member function).
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status ForwardMlaChunk(const FlashMlaAttentionInputs& inputs, int chunk_start, int chunk_end, int batch_size,
                       int max_tokens, int total_prefix_tokens, int max_prefix_per_req, int total_q_tokens,
                       SCALAR_T* q_ptr, SCALAR_T* latent_cache_ptr, SCALAR_T* k_workspace, SCALAR_T* v_workspace,
                       xdnn::Context* xctx, SCALAR_T* kv_cache_layer_ptr, SCALAR_T* output_ptr,
                       const std::shared_ptr<Context>& context_, const std::shared_ptr<Tensor>& workspace_buffer_,
                       int num_heads_, uint32_t qk_nope_head_dim_, uint32_t qk_rope_head_dim_, uint32_t v_head_dim_,
                       uint32_t kv_lora_rank_, int block_token_num_, float attn_scale_);
}  // namespace

Status FlashMlaAttentionLayer::Forward(const FlashMlaAttentionInputs& inputs, Tensor* output) {
  DISPATCH_BY_DTYPE_AND_KVTYPE(inter_data_type_, kv_cache_dtype_, ForwardT, inputs, output);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashMlaAttentionLayer::ForwardT(const FlashMlaAttentionInputs& inputs, Tensor* output) {
  using T = SCALAR_T;

  const Tensor& dp_input_offset_int32 = *inputs.dp_input_offset_int32;
  const Tensor& dp_prefill_q_offset_int32 = *inputs.dp_prefill_q_offset_int32;
  const Tensor& rotary_embedding_pos = *inputs.rotary_embedding_pos;
  const Tensor& forward_shape = *inputs.forward_shape;
  const Tensor& layer_kv_cache = *inputs.layer_kv_cache;
  const Tensor& q_nope_rope_tensor = *inputs.q_nope_rope;
  const Tensor& k_rope_buffer = *inputs.k_rope_buffer;
  const Tensor& k_buffer = *inputs.k_buffer;  // hidden_buffer_1
  const Tensor& v_buffer = *inputs.v_buffer;  // reduce_buffer
  const Tensor& qlod_host_tensor = *inputs.qlod_host_tensor;
  const Tensor& kvlod_host_tensor = *inputs.kvlod_host_tensor;
  const Tensor& layers_slot_mapping = *inputs.layers_slot_mapping;
  const Tensor& atb_attention_attr = *inputs.atb_attention_attr;

  const int batch_size = static_cast<int>(forward_shape.shape[7]);
  const int max_tokens = static_cast<int>(forward_shape.shape[8]);
  const int total_prefix_tokens = static_cast<int>(forward_shape.shape[11]);
  const int max_prefix_per_req = static_cast<int>(forward_shape.shape[13]);
  const int total_q_tokens = q_nope_rope_tensor.shape[0];

  xdnn::Context* xctx = static_cast<xdnn::Context*>(
      context_->device->ext->GetXdnnContext(context_->device->GetPrimaryComputeStream().Get()));

  // ── Step 1 input/output buffer assignment ──────────────────────────────
  // kv_ptr (Step 1 input, kv_b_proj_combined output) and k_workspace (Step 1 output, new k)
  // are assigned to different hidden buffers depending on prefix cache:
  //   With prefix:    kv→hidden_buffer_1, new_k→hidden_buffer_0 (output)
  //   Without prefix: kv→hidden_buffer_0 (output), new_k→hidden_buffer_1
  // This ensures all three constraints are satisfied (see file header).
  T* q_ptr = q_nope_rope_tensor.GetPtr<T>();
  T* latent_cache_ptr = k_rope_buffer.GetPtr<T>();
  int64_t* positions = rotary_embedding_pos.GetPtr<int64_t>();
  T* v_workspace = v_buffer.GetPtr<T>();  // reduce_buffer: new_v (always)

  T* kv_ptr;       // Step 1 input: kv_b_proj_combined output (升维 new kv)
  T* k_workspace;  // Step 1 output: new k
  if (total_prefix_tokens > 0) {
    kv_ptr = k_buffer.GetPtr<T>();      // hidden_buffer_1
    k_workspace = output->GetPtr<T>();  // hidden_buffer_0
  } else {
    kv_ptr = output->GetPtr<T>();        // hidden_buffer_0
    k_workspace = k_buffer.GetPtr<T>();  // hidden_buffer_1
  }

  const float* cos_sin_cache = reinterpret_cast<const float*>(cos_sin_cache_ptr_);

  // ── Step 1: dpsk_prefill_rotary_embedding (full batch) ─────────────────
  // Input:  kv_ptr (升维 kv), q_ptr, latent_cache_ptr (all independent or different buffer)
  // Output: k_workspace (new_k), v_workspace (new_v, reduce_buffer)
  // Constraint 1: kv input ≠ k/v output ✓ (hidden_buffer_0 ≠ hidden_buffer_1 / reduce_buffer)
  const int64_t mla_kv_head_dim = (qk_nope_head_dim_ + v_head_dim_) / 2;
  int ret = XpuDpskPrefillRotaryEmbedding<T, float, int64_t>(
      xctx, kv_ptr, cos_sin_cache, q_ptr, latent_cache_ptr, k_workspace, v_workspace, positions,
      static_cast<int64_t>(num_heads_), static_cast<int64_t>(num_heads_), static_cast<int64_t>(qk_nope_head_dim_),
      mla_kv_head_dim, static_cast<int64_t>(kv_lora_rank_), static_cast<int64_t>(qk_rope_head_dim_),
      static_cast<int64_t>(total_q_tokens));
  if (ret != 0) {
    KLLM_LOG_ERROR << "FlashMlaAttentionLayer: dpsk_prefill_rotary_embedding failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "dpsk_prefill_rotary_embedding failed");
  }

  // ── Step 2: KV cache write (latent_cache roped → paged cache) ──────────
  uint64_t slot_mapping_dim_1 = atb_attention_attr.GetPtr<uint64_t>()[0];
  int32_t* slot_mapping = layers_slot_mapping.GetPtr<int32_t>() + this->layer_index_ * slot_mapping_dim_1;
  T* kv_cache_base = reinterpret_cast<T*>(layer_kv_cache.GetPtr<void*>()[1]);
  T* kv_cache_layer_ptr = reinterpret_cast<T*>(layer_kv_cache.GetPtr<void*>()[1 + this->layer_index_ * 2]);
  int64_t num_blocks = layer_kv_cache.GetPtr<int64_t>()[0];
  int64_t kv_cache_dim = kv_lora_rank_ + qk_rope_head_dim_;

  KLLM_LOG_DEBUG << "FlashMlaAttentionLayer layer=" << this->layer_index_ << " total_q_tokens=" << total_q_tokens
                 << " total_prefix_tokens=" << total_prefix_tokens;

  ret = XpuMlaReshapeCachedKV<T>(xctx, latent_cache_ptr, kv_cache_base, slot_mapping,
                                 static_cast<int64_t>(total_q_tokens), num_blocks, kv_cache_dim,
                                 static_cast<int64_t>(block_token_num_));
  if (ret != 0) {
    KLLM_LOG_ERROR << "FlashMlaAttentionLayer: XpuMlaReshapeCachedKV failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "XpuMlaReshapeCachedKV failed");
  }

  // qlod/kvlod (full batch, for no-prefix path)
  int32_t* qlod = qlod_host_tensor.GetPtr<int32_t>();
  int qlod_len = batch_size + 1;
  int32_t* kvlod = kvlod_host_tensor.GetPtr<int32_t>();
  int kvlod_len = batch_size + 1;
  const int32_t* qlod_dev = dp_prefill_q_offset_int32.GetPtr<int32_t>();
  const int32_t* kvlod_dev = dp_input_offset_int32.GetPtr<int32_t>();

  float prefill_alpha = attn_scale_ * std::sqrt(static_cast<float>(qk_nope_head_dim_ + qk_rope_head_dim_));

  if (total_prefix_tokens == 0) {
    // ── Step 2.5: No prefix cache — single self-attention ─────────────────
    // Input:  q_ptr (independent), new_k (hidden_buffer_1), new_v (reduce_buffer)
    // Output: hidden_buffer_0 (output)
    // Constraint 2: k/v input (hidden_buffer_1/reduce_buffer) ≠ output (hidden_buffer_0) ✓
    ret = XpuFlashMlaAttention<T, int32_t>(
        xctx, q_ptr, k_workspace, v_workspace, output->GetPtr<T>(), static_cast<int64_t>(batch_size),
        static_cast<int64_t>(max_tokens), static_cast<int64_t>(max_tokens), static_cast<int64_t>(num_heads_),
        static_cast<int64_t>(num_heads_), static_cast<int64_t>(qk_nope_head_dim_ + qk_rope_head_dim_),
        static_cast<int64_t>(v_head_dim_), is_causal_, prefill_alpha, qlod, qlod_len, kvlod, kvlod_len,
        static_cast<int64_t>(block_token_num_), qlod_dev, kvlod_dev);
    if (ret != 0) {
      KLLM_LOG_ERROR << "FlashMlaAttentionLayer: XpuFlashMlaAttention failed, ret=" << ret;
      return Status(RET_RUNTIME_FAILED, "XpuFlashMlaAttention failed");
    }
  } else {
    // ── Prefix cache path: chunked processing ─────────────────────────────
    const Tensor& chunk_kv_batch_starts_host = *inputs.chunk_kv_batch_starts_host;
    const int* chunk_kv_batch_starts_ptr = chunk_kv_batch_starts_host.GetPtr<int>();
    const size_t num_chunks =
        chunk_kv_batch_starts_host.shape[0] > 0 ? static_cast<size_t>(chunk_kv_batch_starts_host.shape[0]) - 1 : 0;

    if (num_chunks == 0) {
      // Fallback: single chunk = entire batch
      STATUS_CHECK_RETURN((ForwardMlaChunk<SCALAR_T, CACHE_T, KV_DTYPE>(
          inputs, 0, batch_size, batch_size, max_tokens, total_prefix_tokens, max_prefix_per_req, total_q_tokens, q_ptr,
          latent_cache_ptr, k_workspace, v_workspace, xctx, kv_cache_layer_ptr, output->GetPtr<T>(), context_,
          workspace_buffer_, num_heads_, qk_nope_head_dim_, qk_rope_head_dim_, v_head_dim_, kv_lora_rank_,
          block_token_num_, attn_scale_)));
    } else {
      for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const int chunk_start = chunk_kv_batch_starts_ptr[chunk_idx];
        const int chunk_end = chunk_kv_batch_starts_ptr[chunk_idx + 1];
        STATUS_CHECK_RETURN((ForwardMlaChunk<SCALAR_T, CACHE_T, KV_DTYPE>(
            inputs, chunk_start, chunk_end, batch_size, max_tokens, total_prefix_tokens, max_prefix_per_req,
            total_q_tokens, q_ptr, latent_cache_ptr, k_workspace, v_workspace, xctx, kv_cache_layer_ptr,
            output->GetPtr<T>(), context_, workspace_buffer_, num_heads_, qk_nope_head_dim_, qk_rope_head_dim_,
            v_head_dim_, kv_lora_rank_, block_token_num_, attn_scale_)));
      }
    }
  }

  output->shape = {static_cast<size_t>(total_q_tokens), static_cast<size_t>(num_heads_ * v_head_dim_)};
  return Status();
}

namespace {
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status ForwardMlaChunk(const FlashMlaAttentionInputs& inputs, int chunk_start, int chunk_end, int batch_size,
                       int max_tokens, int total_prefix_tokens, int max_prefix_per_req, int total_q_tokens,
                       SCALAR_T* q_ptr, SCALAR_T* latent_cache_ptr, SCALAR_T* k_workspace, SCALAR_T* v_workspace,
                       xdnn::Context* xctx, SCALAR_T* kv_cache_layer_ptr, SCALAR_T* output_ptr,
                       const std::shared_ptr<Context>& context_, const std::shared_ptr<Tensor>& workspace_buffer_,
                       int num_heads_, uint32_t qk_nope_head_dim_, uint32_t qk_rope_head_dim_, uint32_t v_head_dim_,
                       uint32_t kv_lora_rank_, int block_token_num_, float attn_scale_) {
  using T = SCALAR_T;
  int ret = 0;

  // ── Chunk-scoped offsets ───────────────────────────────────────────────
  const size_t* dp_prefill_q_offset_uint64_ptr = inputs.dp_prefill_q_offset_host->GetPtr<size_t>();

  const size_t chunk_q_tokens_offset = dp_prefill_q_offset_uint64_ptr[chunk_start];
  const size_t chunk_q_tokens = dp_prefill_q_offset_uint64_ptr[chunk_end] - chunk_q_tokens_offset;
  // BUGFIX: inputs.dp_input_prefix aliases dp_input_prefix_uint64_tensor, which is LOCATION_DEVICE.
  // Dereferencing it on the host (as the old code did) reads garbage prefix token counts, which
  // then corrupt every workspace offset, the GEMM M dim, and the gather bounds below → NaN/OOB.
  // Use the host int32 copy (prefix_kvlod_host_tensor == dp_input_prefix_int32_host_tensor), which
  // holds the identical cumulative prefix lengths [0, p0, p0+p1, ...] — the same array indexed at
  // line ~333 for chunk_prefix_kvlod.
  const int32_t* prefix_kvlod_host_ptr = inputs.prefix_kvlod_host_tensor->GetPtr<int32_t>();
  const size_t chunk_prefix_tokens_offset = static_cast<size_t>(prefix_kvlod_host_ptr[chunk_start]);
  const size_t chunk_prefix_tokens = static_cast<size_t>(prefix_kvlod_host_ptr[chunk_end]) - chunk_prefix_tokens_offset;
  const int chunk_batch_size = chunk_end - chunk_start;

  // Build chunk-local qlod/kvlod/prefix_kvlod (shift to 0-based)
  const Tensor& qlod_host_tensor = *inputs.qlod_host_tensor;
  const Tensor& kvlod_host_tensor = *inputs.kvlod_host_tensor;
  const Tensor& prefix_kvlod_host_tensor = *inputs.prefix_kvlod_host_tensor;
  const int32_t* qlod = qlod_host_tensor.GetPtr<int32_t>();
  const int32_t* kvlod = kvlod_host_tensor.GetPtr<int32_t>();
  const int32_t* prefix_kvlod_full = prefix_kvlod_host_tensor.GetPtr<int32_t>();

  std::vector<int32_t> chunk_qlod(chunk_batch_size + 1);
  std::vector<int32_t> chunk_kvlod(chunk_batch_size + 1);
  std::vector<int32_t> chunk_prefix_kvlod(chunk_batch_size + 1);
  for (int i = 0; i <= chunk_batch_size; ++i) {
    chunk_qlod[i] = qlod[chunk_start + i] - qlod[chunk_start];
    chunk_kvlod[i] = kvlod[chunk_start + i] - kvlod[chunk_start];
    chunk_prefix_kvlod[i] = prefix_kvlod_full[chunk_start + i] - prefix_kvlod_full[chunk_start];
  }

  // Device lod pointers (offset into full device array)
  const Tensor& dp_prefill_q_offset_int32 = *inputs.dp_prefill_q_offset_int32;
  const Tensor& dp_input_offset_int32 = *inputs.dp_input_offset_int32;
  const Tensor& prefix_kvlod_dev_tensor = *inputs.prefix_kvlod_dev_tensor;
  const int32_t* chunk_qlod_dev = dp_prefill_q_offset_int32.GetPtr<int32_t>() + chunk_start;
  const int32_t* chunk_kvlod_dev = dp_input_offset_int32.GetPtr<int32_t>() + chunk_start;
  const int32_t* chunk_prefix_kvlod_dev = prefix_kvlod_dev_tensor.GetPtr<int32_t>() + chunk_start;

  const int64_t qk_head_dim = qk_nope_head_dim_ + qk_rope_head_dim_;
  const size_t q_per_token_size = static_cast<size_t>(num_heads_) * qk_head_dim;

  // Chunk-local Q pointer (independent buffer, offset by chunk_q_tokens_offset)
  T* chunk_q_ptr = q_ptr + chunk_q_tokens_offset * q_per_token_size;

  const int chunk_max_seq_q = chunk_qlod[chunk_batch_size];
  const int chunk_max_prefix_per_req = chunk_prefix_kvlod[chunk_batch_size];
  float prefill_alpha = attn_scale_ * std::sqrt(static_cast<float>(qk_head_dim));

  KLLM_LOG_DEBUG << "ForwardChunkT chunk=[" << chunk_start << "," << chunk_end << ")"
                 << " batch=" << chunk_batch_size << " q_tokens=" << chunk_q_tokens
                 << " prefix_tokens=" << chunk_prefix_tokens;

  // ── Buffer pointers (chunk-local offsets) ──────────────────────────────
  // new_k: hidden_buffer_0, from Step 1 (prefix path: k_workspace = output_ptr)
  T* chunk_new_k = k_workspace + chunk_q_tokens_offset * num_heads_ * qk_head_dim;
  // new_v: reduce_buffer, from Step 1
  T* chunk_new_v = v_workspace + chunk_q_tokens_offset * num_heads_ * v_head_dim_;

  // o_new / o_prefix: hidden_buffer_1 (k_buffer), Step 1 input (kv_b_proj output) freed
  const Tensor& k_buffer = *inputs.k_buffer;
  T* o_new = k_buffer.GetPtr<T>() + chunk_q_tokens_offset * num_heads_ * v_head_dim_;
  T* o_prefix = o_new + chunk_q_tokens * num_heads_ * v_head_dim_;

  // ── layer_workspace layout (sequential, per chunk) ─────────────────────
  // [prefix_kv_c][prefix_k_pe][prefix_kv_proj][prefix_k][prefix_v]
  T* ws = workspace_buffer_->GetPtr<T>();
  T* prefix_kv_c = ws;                                        // Phase 3 output: [T_p_ch * kv_lora_rank]
  T* prefix_k_pe = ws + chunk_prefix_tokens * kv_lora_rank_;  // Phase 3 output: [T_p_ch * qk_rope]
  T* prefix_kv_proj =                                         // Phase 4 output: [T_p_ch * H*(qk_nope+v_head)]
      ws + chunk_prefix_tokens * (kv_lora_rank_ + qk_rope_head_dim_);
  T* prefix_k =  // Phase 5 output: [T_p_ch * H*qk_head_dim] (gap at qk_nope for k_pe)
      ws + chunk_prefix_tokens * (kv_lora_rank_ + qk_rope_head_dim_ + num_heads_ * (qk_nope_head_dim_ + v_head_dim_));
  T* prefix_v = prefix_k + chunk_prefix_tokens * num_heads_ * qk_head_dim;  // Phase 5 output: [T_p_ch * H*v_head]

  // ── Phase 3: Gather prefix compressed KV from paged cache ──────────────
  // Input:  paged cache blocks (kv_cache_layer_ptr)
  // Output: layer_workspace prefix_kv_c [T_p_ch * kv_lora_rank]
  //         layer_workspace prefix_k_pe [T_p_ch * qk_rope]
  const Tensor& atb_attention_attr = *inputs.atb_attention_attr;
  const Tensor& block_table = *inputs.block_table;
  const Tensor& kv_b_proj_combined_weight = *inputs.kv_b_proj_combined_weight;
  uint64_t max_num_blocks_per_query = atb_attention_attr.GetPtr<uint64_t>()[1];
  const int32_t* block_table_ptr = block_table.GetPtr<int32_t>() + chunk_start * max_num_blocks_per_query;

  ret = XpuGatherMlaCache<T, int32_t>(
      xctx, kv_cache_layer_ptr, prefix_kv_c, prefix_k_pe, static_cast<int64_t>(kv_lora_rank_),
      static_cast<int64_t>(qk_rope_head_dim_), chunk_prefix_kvlod.data(), static_cast<int>(chunk_prefix_kvlod.size()),
      /*seq_lod_device=*/nullptr, block_table_ptr, static_cast<int64_t>(block_token_num_),
      static_cast<int64_t>(max_num_blocks_per_query));
  if (ret != 0) {
    KLLM_LOG_ERROR << "FlashMlaAttentionLayer: XpuGatherMlaCache failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "XpuGatherMlaCache failed");
  }

  // ── Phase 4: kv_b_proj for prefix (prefix_kv_c → prefix_kv_proj) ───────
  // Input:  layer_workspace prefix_kv_c [T_p_ch * kv_lora_rank]
  // Output: layer_workspace prefix_kv_proj [T_p_ch * H * (qk_nope + v_head)]
  // (prefix_kv_c freed after matmul)
  {
    const int64_t M = chunk_prefix_tokens;
    const int64_t K = kv_lora_rank_;
    const int64_t N = num_heads_ * (qk_nope_head_dim_ + v_head_dim_);

    if (kv_b_proj_combined_weight.weight_scales != nullptr) {
      ret = XpuInt8ScaledMM<T>(xctx, prefix_kv_c, kv_b_proj_combined_weight.GetPtr<int8_t>(),
                               kv_b_proj_combined_weight.weight_scales->GetPtr<float>(), prefix_kv_proj, M, N, K);
    } else {
      ret = XpuFcFusion<T, T, T>(xctx, prefix_kv_c, kv_b_proj_combined_weight.GetPtr<T>(), prefix_kv_proj, M, N, K,
                                 /*x_trans=*/false, /*w_trans=*/true,
                                 /*x_maxptr=*/nullptr, /*w_maxptr=*/nullptr, /*y_maxptr=*/nullptr);
    }
    if (ret != 0) {
      KLLM_LOG_ERROR << "FlashMlaAttentionLayer: prefix kv_b_proj failed, ret=" << ret;
      return Status(RET_RUNTIME_FAILED, "prefix kv_b_proj failed");
    }
  }

  // ── Phase 5: Split prefix_kv_proj → prefix_k (k_nope) + prefix_v ───────
  // Input:  layer_workspace prefix_kv_proj [T_p_ch * H * (qk_nope + v_head)]
  // Output: layer_workspace prefix_k [T_p_ch * H * qk_head_dim]  (only k_nope filled, gap at [qk_nope:qk_head_dim])
  //         layer_workspace prefix_v [T_p_ch * H * v_head_dim]
  // (prefix_kv_proj freed after split)
  {
    auto* xpuctx = static_cast<xpukernel::Context*>(
        context_->device->ext->GetXpuKernelContext(context_->device->GetPrimaryComputeStream().Get()));
    int64_t outer = static_cast<int64_t>(chunk_prefix_tokens) * num_heads_;
    int64_t src_stride = qk_nope_head_dim_ + v_head_dim_;

    // Copy k_nope columns [0:qk_nope] from prefix_kv_proj into prefix_k[:, 0:qk_nope]
    ret = XpuCopy2d<T>(xpuctx, prefix_kv_proj, prefix_k, outer, src_stride, qk_head_dim, qk_nope_head_dim_);
    if (ret != 0) {
      KLLM_LOG_ERROR << "FlashMlaAttentionLayer: split k_nope failed, ret=" << ret;
      return Status(RET_RUNTIME_FAILED, "split k_nope failed");
    }

    // Copy v columns [qk_nope:qk_nope+v_head] from prefix_kv_proj into prefix_v
    ret =
        XpuCopy2d<T>(xpuctx, prefix_kv_proj + qk_nope_head_dim_, prefix_v, outer, src_stride, v_head_dim_, v_head_dim_);
    if (ret != 0) {
      KLLM_LOG_ERROR << "FlashMlaAttentionLayer: split prefix_v failed, ret=" << ret;
      return Status(RET_RUNTIME_FAILED, "split prefix_v failed");
    }
  }

  // ── Phase 6: k_pe expand (prefix_k_pe → fill prefix_k gap) ─────────────
  // Input:  layer_workspace prefix_k_pe [T_p_ch * qk_rope]
  // Output: fills prefix_k[:, qk_nope:qk_head_dim] (broadcast over heads)
  // (prefix_k_pe freed after expand)
  ret = XpuKpeExpand<T>(xctx, prefix_k_pe, prefix_k, static_cast<int64_t>(chunk_prefix_tokens),
                        static_cast<int64_t>(num_heads_), /*v_head_dim=*/0, static_cast<int64_t>(qk_nope_head_dim_),
                        static_cast<int64_t>(qk_rope_head_dim_));
  if (ret != 0) {
    KLLM_LOG_ERROR << "FlashMlaAttentionLayer: XpuKpeExpand failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "XpuKpeExpand failed");
  }

  // ── Phase 7: Self-attention with LSE (new Q × new K/V, causal) ────────
  // Input:  chunk_q_ptr (independent), chunk_new_k (hidden_buffer_0), chunk_new_v (reduce_buffer)
  // Output: o_new → hidden_buffer_1 [T_q_ch * H * v_head]
  //         lse_new → latent_cache [H * T_q_ch] (float, offset = chunk_q_tokens_offset * H)
  // Constraint 2: k/v input (hidden_buffer_0/reduce_buffer) ≠ o_new output (hidden_buffer_1) ✓
  float* lse_new = reinterpret_cast<float*>(latent_cache_ptr) + chunk_q_tokens_offset * num_heads_;

  ret = XpuFlashMlaAttentionWithLse<T, int32_t>(
      xctx, chunk_q_ptr, chunk_new_k, chunk_new_v, o_new, lse_new, /*unpadded_lse=*/true,
      static_cast<int64_t>(chunk_batch_size), static_cast<int64_t>(chunk_max_seq_q),
      static_cast<int64_t>(chunk_max_seq_q), static_cast<int64_t>(num_heads_), static_cast<int64_t>(num_heads_),
      qk_head_dim, static_cast<int64_t>(v_head_dim_),
      /*is_causal=*/true, prefill_alpha, chunk_qlod.data(), static_cast<int>(chunk_qlod.size()), chunk_qlod.data(),
      static_cast<int>(chunk_qlod.size()), static_cast<int64_t>(block_token_num_), chunk_qlod_dev, chunk_qlod_dev);
  if (ret != 0) {
    KLLM_LOG_ERROR << "FlashMlaAttentionLayer: self-attention with LSE failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "self-attention with LSE failed");
  }

  // ── Phase 8: Cross-attention with LSE (new Q × prefix K/V, non-causal) ─
  // Input:  chunk_q_ptr (independent), prefix_k/prefix_v (layer_workspace)
  // Output: o_prefix → hidden_buffer_1 [T_q_ch * H * v_head] (after o_new)
  //         lse_prefix → latent_cache [H * T_q_ch] (float, after lse_new)
  float* lse_prefix = lse_new + chunk_q_tokens * num_heads_;

  ret = XpuFlashMlaAttentionWithLse<T, int32_t>(
      xctx, chunk_q_ptr, prefix_k, prefix_v, o_prefix, lse_prefix, /*unpadded_lse=*/true,
      static_cast<int64_t>(chunk_batch_size), static_cast<int64_t>(chunk_max_seq_q),
      static_cast<int64_t>(chunk_max_prefix_per_req), static_cast<int64_t>(num_heads_),
      static_cast<int64_t>(num_heads_), qk_head_dim, static_cast<int64_t>(v_head_dim_),
      /*is_causal=*/false, prefill_alpha, chunk_qlod.data(), static_cast<int>(chunk_qlod.size()),
      chunk_prefix_kvlod.data(), static_cast<int>(chunk_prefix_kvlod.size()), static_cast<int64_t>(block_token_num_),
      chunk_qlod_dev, chunk_prefix_kvlod_dev);
  if (ret != 0) {
    KLLM_LOG_ERROR << "FlashMlaAttentionLayer: cross-attention with LSE failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "cross-attention with LSE failed");
  }

  // ── Phase 9: Permute LSE [H, T_q_ch] → [T_q_ch, H] ────────────────────
  // Input/Output: latent_cache (chunk-local, 4 regions of H*T_q_ch floats each)
  //   [lse_new:      H*T_q_ch]  → [lse_new_T:      T_q_ch*H]
  //   [lse_prefix:   H*T_q_ch]  → [lse_prefix_T:   T_q_ch*H]
  const int64_t lse_elems = static_cast<int64_t>(chunk_q_tokens) * num_heads_;
  float* lse_new_T = lse_new + 2 * lse_elems;
  float* lse_prefix_T = lse_new + 3 * lse_elems;

  Stream& permute_stream = context_->device->GetPrimaryComputeStream();
  {
    Tensor lse_new_in(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
                      {static_cast<size_t>(num_heads_), static_cast<size_t>(chunk_q_tokens)}, lse_new, "lse_new_in");
    Tensor lse_new_out(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
                       {static_cast<size_t>(chunk_q_tokens), static_cast<size_t>(num_heads_)}, lse_new_T,
                       "lse_new_out");
    STATUS_CHECK_RETURN(Permute(lse_new_in, lse_new_out, {1, 0}, permute_stream, xctx));
  }
  {
    Tensor lse_prefix_in(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
                         {static_cast<size_t>(num_heads_), static_cast<size_t>(chunk_q_tokens)}, lse_prefix,
                         "lse_prefix_in");
    Tensor lse_prefix_out(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
                          {static_cast<size_t>(chunk_q_tokens), static_cast<size_t>(num_heads_)}, lse_prefix_T,
                          "lse_prefix_out");
    STATUS_CHECK_RETURN(Permute(lse_prefix_in, lse_prefix_out, {1, 0}, permute_stream, xctx));
  }

  // ── Phase 10: Merge attention outputs ──────────────────────────────────
  // Input:  o_new (hidden_buffer_1, first half), o_prefix (hidden_buffer_1, second half)
  //         lse_new_T, lse_prefix_T (latent_cache)
  // Output: final output → hidden_buffer_0 [T_q_ch * H * v_head] (new_k freed after Phase 7)
  // Constraint 3: o_new/o_prefix input (hidden_buffer_1) ≠ output (hidden_buffer_0) ✓
  T* chunk_output = output_ptr + chunk_q_tokens_offset * num_heads_ * v_head_dim_;
  ret = XpuAttentionMergeState<T>(xctx, o_new, lse_new_T, o_prefix, lse_prefix_T, chunk_output,
                                  /*s_merged=*/nullptr, static_cast<int64_t>(chunk_q_tokens),
                                  static_cast<int64_t>(num_heads_), static_cast<int64_t>(v_head_dim_));
  if (ret != 0) {
    KLLM_LOG_ERROR << "FlashMlaAttentionLayer: attention_merge_state failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "attention_merge_state failed");
  }

  return Status();
}
}  // namespace

// Explicit instantiations for fp16 and bf16. KV_DTYPE=kAuto is the only path on Kunlun.
// ForwardMlaChunk (file-local) instantiates implicitly through these ForwardT instantiations.
template Status FlashMlaAttentionLayer::ForwardT<float16, float16, llm_kernels::utils::KVCacheType::kAuto>(
    const FlashMlaAttentionInputs&, Tensor*);
template Status FlashMlaAttentionLayer::ForwardT<bfloat16, bfloat16, llm_kernels::utils::KVCacheType::kAuto>(
    const FlashMlaAttentionInputs&, Tensor*);

}  // namespace ksana_llm
