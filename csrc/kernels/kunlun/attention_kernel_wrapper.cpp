/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "kernels/kunlun/attention_kernel_wrapper.h"

// XPU 基础头单独成块，确保排在 xfa/infer_ops 之前（.clang-format 里 IncludeBlocks: Preserve，
// 块内排序、块间不重排）。flash_api.h 来自 XMLIR SDK 且未验证自包含，不要与本块合并。
#include <xpu/runtime.h>
#include <xpu/xdnn.h>

#include <flash_api.h>       // xfa::flash_attention_context_vllm, xfa::paged_attention
#include <infer_ops.h>       // infer_ops::reshape_and_cache
#include <infer_ops_rope.h>  // infer_ops::rotary_embedding_neox

#include <vector>

#include "utils/logger.h"

namespace xfa = baidu::xpu::xfa;

namespace ksana_llm {

// ── RoPE ─────────────────────────────────────────────────────────────────────

template <typename T>
int XpuRotaryEmbeddingNeox(xdnn::Context* ctx, const int64_t* positions, T* query, T* key, const T* cos_sin_cache,
                           int rot_dim, int num_heads, int num_kv_heads, int head_size, int num_tokens) {
  return infer_ops::rotary_embedding_neox<xdnn_t<T>, xdnn_t<T>, int64_t>(
      ctx, positions, xcast_m(query), xcast_m(key), xcast(cos_sin_cache), static_cast<int64_t>(rot_dim),
      static_cast<int64_t>(num_heads), static_cast<int64_t>(num_kv_heads), static_cast<int64_t>(head_size),
      static_cast<int64_t>(num_tokens));
}

template int XpuRotaryEmbeddingNeox<float16>(xdnn::Context*, const int64_t*, float16*, float16*, const float16*, int,
                                             int, int, int, int);
template int XpuRotaryEmbeddingNeox<bfloat16>(xdnn::Context*, const int64_t*, bfloat16*, bfloat16*, const bfloat16*,
                                              int, int, int, int, int);

// ── Fused Split + Norm + RoPE ────────────────────────────────────────────────

template <typename T>
int XpuSplitNormRopeNeox(xdnn::Context* ctx, const T* qkv, const T* rotary_pos_emb, const T* q_norm_weight,
                         const T* k_norm_weight, const int64_t* positions, T* q_out, T* k_out, T* v_out,
                         int64_t num_tokens, int64_t max_seqlen, int64_t head_num, int64_t kv_head_num,
                         int64_t head_dim, int64_t rotary_dim, float eps) {
  return infer_ops::split_norm_rope_neox<xdnn_t<T>, int64_t>(
      ctx, xcast(qkv), xcast(rotary_pos_emb), xcast(q_norm_weight), xcast(k_norm_weight), positions, xcast_m(q_out),
      xcast_m(k_out), xcast_m(v_out), num_tokens, /*emb_batch_size=*/1, max_seqlen, head_num, kv_head_num, head_dim,
      rotary_dim,
      /*cache_align_to_gpu=*/false,
      /*t_value=*/-1, /*h_value=*/-1, /*w_value=*/-1,
      /*mrope_use_interleave=*/false, eps);
}

template int XpuSplitNormRopeNeox<float16>(xdnn::Context*, const float16*, const float16*, const float16*,
                                           const float16*, const int64_t*, float16*, float16*, float16*, int64_t,
                                           int64_t, int64_t, int64_t, int64_t, int64_t, float);
template int XpuSplitNormRopeNeox<bfloat16>(xdnn::Context*, const bfloat16*, const bfloat16*, const bfloat16*,
                                            const bfloat16*, const int64_t*, bfloat16*, bfloat16*, bfloat16*, int64_t,
                                            int64_t, int64_t, int64_t, int64_t, int64_t, float);

// ── Fused Split + RoPE (no QK-Norm) ──────────────────────────────────────────

template <typename T>
int XpuSplitRopeNeox(xdnn::Context* ctx, const T* qkv, const T* cos_sin_cache, const int64_t* positions, T* q_out,
                     T* k_out, T* v_out, int64_t num_tokens, int64_t max_seqlen, int64_t head_num, int64_t kv_head_num,
                     int64_t head_dim, int64_t rotary_dim) {
  return infer_ops::split_rotary_embedding<xdnn_t<T>, xdnn_t<T>, int64_t>(
      ctx, xcast(qkv), xcast(cos_sin_cache), positions, xcast_m(q_out), xcast_m(k_out), xcast_m(v_out), num_tokens,
      max_seqlen, head_num, kv_head_num, head_dim, rotary_dim,
      /*is_neox=*/true);
}

template int XpuSplitRopeNeox<float16>(xdnn::Context*, const float16*, const float16*, const int64_t*, float16*,
                                       float16*, float16*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
template int XpuSplitRopeNeox<bfloat16>(xdnn::Context*, const bfloat16*, const bfloat16*, const int64_t*, bfloat16*,
                                        bfloat16*, bfloat16*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

// ── Flash Attention Context ───────────────────────────────────────────────────

template <typename T, typename TCACHE, typename TID>
int XpuFlashAttentionContext(xdnn::Context* ctx, const T* q, const T* k, const T* v, const TCACHE* k_cache,
                             const TCACHE* v_cache, T* output, int64_t batch_num, int64_t max_seq_q, int64_t max_seq_kv,
                             int64_t max_seq_cache, int64_t head_num, int64_t kv_head_num, int64_t head_dim,
                             bool is_causal, float softmax_scale, const TID* qlod, int qlod_len, const TID* kvlod,
                             int kvlod_len, const TID* cachelod, int cachelod_len, const TID* block_table,
                             int64_t block_size, int64_t max_blocks_per_seq) {
  using xfa_t = xdnn_t<T>;
  using xfa_cache_t = xdnn_t<TCACHE>;

  xdnn::VectorParam<TID> qlod_param = {qlod, qlod_len, nullptr};
  xdnn::VectorParam<TID> kvlod_param = {kvlod, kvlod_len, nullptr};
  std::vector<int64_t> mask_bias_shape = {};
  std::vector<int64_t> alibi_slopes_shape = {};
  int64_t vo_head_dim = head_dim;

  // Compute q_maxptr via xdnn::findmax (matches xsglang klx_attention_extend.cc reference).
  float* q_maxptr_dev = nullptr;
  void* q_max_buffer = nullptr;
  int64_t total_tokens_q = (qlod != nullptr && qlod_len >= 2) ? qlod[qlod_len - 1] : (batch_num * max_seq_q);
  int64_t q_total_elems = total_tokens_q * head_num * head_dim;
  size_t maxptr_bytes = static_cast<size_t>(ctx->max_ptr_size()) * sizeof(float);
  if (xpu_malloc(&q_max_buffer, maxptr_bytes) == 0 && q_max_buffer != nullptr) {
    q_maxptr_dev = reinterpret_cast<float*>(q_max_buffer);
    if (xdnn::findmax<xfa_t>(ctx, reinterpret_cast<const xfa_t*>(q), q_maxptr_dev, q_total_elems) != 0) {
      xpu_free(q_max_buffer);
      q_max_buffer = nullptr;
      q_maxptr_dev = nullptr;
    }
  }

  // Template args: <T_Q, T_KV, T_KV_C, T_BIAS, T_O, T_GEMM0, T_GEMM1, T_EW, TID, quant_mode>
  // fp16 variant: all-fp16 (verified working).
  int ret = xfa::flash_attention_context_vllm<xfa_t, xfa_t, xfa_cache_t, xfa_t, xfa_t, xfa_t, xfa_t, xfa_t, TID, 0>(
      ctx, reinterpret_cast<const xfa_t*>(q), reinterpret_cast<const xfa_t*>(k), reinterpret_cast<const xfa_t*>(v),
      /*mask_bias=*/nullptr, reinterpret_cast<xfa_t*>(output), batch_num, max_seq_q, max_seq_kv, head_num, head_dim,
      kv_head_num,
      /*alibi_slopes=*/nullptr, qlod_param, kvlod_param, mask_bias_shape, alibi_slopes_shape, is_causal,
      /*is_qkv_fusion=*/false,
      /*qkv_layout=*/0x0010,  // ATTN_BLHD
      /*vsl_type=*/0,         // ATTN_NOVSL
      /*q_maxptr=*/q_maxptr_dev, /*k_maxptr=*/nullptr, /*v_maxptr=*/nullptr, /*o_maxptr=*/nullptr, vo_head_dim,
      reinterpret_cast<const xfa_cache_t*>(k_cache), reinterpret_cast<const xfa_cache_t*>(v_cache),
      /*k_cache_maxptr=*/nullptr, /*v_cache_maxptr=*/nullptr, block_size, max_blocks_per_seq, /*prefill_len=*/-1,
      block_table, softmax_scale, /*swa_left_size=*/-1, /*swa_right_size=*/-1,
      /*softmax_lse=*/nullptr, /*unpadded_lse=*/false, /*attn_sink=*/nullptr);

  if (q_max_buffer != nullptr) xpu_free(q_max_buffer);
  return ret;
}

template int XpuFlashAttentionContext<float16, float16, int32_t>(xdnn::Context*, const float16*, const float16*,
                                                                 const float16*, const float16*, const float16*,
                                                                 float16*, int64_t, int64_t, int64_t, int64_t, int64_t,
                                                                 int64_t, int64_t, bool, float, const int32_t*, int,
                                                                 const int32_t*, int, const int32_t*, int,
                                                                 const int32_t*, int64_t, int64_t);

// bf16: align with xsglang FA_CONTEXT<bfloat16> — T_BIAS=float, T_GEMM0/1=float, T_EW=float
// (full fp32 accumulator; "tfloat32 miss accuracy in xhpc0204" per xsglang comment).
template <>
int XpuFlashAttentionContext<bfloat16, bfloat16, int32_t>(
    xdnn::Context* ctx, const bfloat16* q, const bfloat16* k, const bfloat16* v, const bfloat16* k_cache,
    const bfloat16* v_cache, bfloat16* output, int64_t batch_num, int64_t max_seq_q, int64_t max_seq_kv,
    int64_t max_seq_cache, int64_t head_num, int64_t kv_head_num, int64_t head_dim, bool is_causal, float softmax_scale,
    const int32_t* qlod, int qlod_len, const int32_t* kvlod, int kvlod_len, const int32_t* /*cachelod*/,
    int /*cachelod_len*/, const int32_t* block_table, int64_t block_size, int64_t max_blocks_per_seq) {
  xdnn::VectorParam<int32_t> qlod_param = {qlod, qlod_len, nullptr};
  xdnn::VectorParam<int32_t> kvlod_param = {kvlod, kvlod_len, nullptr};
  std::vector<int64_t> mask_bias_shape = {};
  std::vector<int64_t> alibi_slopes_shape = {};
  int64_t vo_head_dim = head_dim;

  float* q_maxptr_dev = nullptr;
  void* q_max_buffer = nullptr;
  int64_t total_tokens_q = (qlod != nullptr && qlod_len >= 2) ? qlod[qlod_len - 1] : (batch_num * max_seq_q);
  int64_t q_total_elems = total_tokens_q * head_num * head_dim;
  size_t maxptr_bytes = static_cast<size_t>(ctx->max_ptr_size()) * sizeof(float);
  if (xpu_malloc(&q_max_buffer, maxptr_bytes) == 0 && q_max_buffer != nullptr) {
    q_maxptr_dev = reinterpret_cast<float*>(q_max_buffer);
    if (xdnn::findmax<xdnn_bf16>(ctx, xcast(q), q_maxptr_dev, q_total_elems) != 0) {
      xpu_free(q_max_buffer);
      q_max_buffer = nullptr;
      q_maxptr_dev = nullptr;
    }
  }

  int ret = xfa::flash_attention_context_vllm<xdnn_bf16, xdnn_bf16, xdnn_bf16, float, xdnn_bf16, float, float, float,
                                              int32_t, 0>(
      ctx, xcast(q), xcast(k), xcast(v),
      /*mask_bias=*/nullptr, xcast_m(output), batch_num, max_seq_q, max_seq_kv, head_num, head_dim, kv_head_num,
      /*alibi_slopes=*/nullptr, qlod_param, kvlod_param, mask_bias_shape, alibi_slopes_shape, is_causal,
      /*is_qkv_fusion=*/false,
      /*qkv_layout=*/0x0010, /*vsl_type=*/0,
      /*q_maxptr=*/q_maxptr_dev, nullptr, nullptr, nullptr, vo_head_dim, xcast(k_cache), xcast(v_cache), nullptr,
      nullptr, block_size, max_blocks_per_seq, /*prefill_len=*/-1, block_table, softmax_scale, /*swa_left_size=*/-1,
      /*swa_right_size=*/-1,
      /*softmax_lse=*/nullptr, /*unpadded_lse=*/false, /*attn_sink=*/nullptr);

  (void)max_seq_cache;
  if (q_max_buffer != nullptr) xpu_free(q_max_buffer);
  return ret;
}

// fp32 stub: DISPATCH_BY_DTYPE_AND_KVTYPE expands TYPE_FP32 unconditionally; never called at runtime.
template <>
int XpuFlashAttentionContext<float, float, int32_t>(xdnn::Context*, const float*, const float*, const float*,
                                                    const float*, const float*, float*, int64_t, int64_t, int64_t,
                                                    int64_t, int64_t, int64_t, int64_t, bool, float, const int32_t*,
                                                    int, const int32_t*, int, const int32_t*, int, const int32_t*,
                                                    int64_t, int64_t) {
  KLLM_LOG_ERROR << "XpuFlashAttentionContext<float>: fp32 attention not supported on Kunlun";
  return -1;
}

// ── KV Cache Scatter ──────────────────────────────────────────────────────────

// NOTE ON THE SLOT-MAPPING INDEX TYPE — do NOT "align" this with xsglang.
//
// xsglang instantiates reshape_and_cache<bfloat16, long> while we use
// reshape_and_cache<bfloat16, int32_t>. A trace comparison of the two on the
// same workload shows the int32_t instantiation is dramatically faster:
//
//   ksana    reshape_and_cache<bfloat16, int>   28043 calls /   30.3ms  (1.08us avg)
//   xsglang  reshape_and_cache<bfloat16, long>  20520 calls / 5761.5ms  (280.77us avg)
//
// i.e. ~260x faster per call, and the single largest win we hold over xsglang.
// The int64_t instantiation appears to miss the vectorised scatter path.
//
// int32_t is also safe for addressing: slot ids are bounded by
// num_blocks * block_size (total KV slots), far below INT32_MAX for any
// configuration we can actually allocate on device.
template <typename T>
int XpuReshapeCachedKV(xdnn::Context* ctx, const T* k, const T* v, T* k_cache, T* v_cache, int32_t* slot_mapping,
                       int64_t num_tokens, int64_t num_blocks, int64_t num_heads, int64_t head_size,
                       int64_t block_size) {
  KLLM_LOG_DEBUG << "XpuReshapeCachedKV: tokens=" << num_tokens << " blocks=" << num_blocks << " heads=" << num_heads
                 << " head_size=" << head_size << " block_size=" << block_size;
  int ret = infer_ops::reshape_and_cache<xdnn_t<T>, int32_t>(
      ctx, reinterpret_cast<const xdnn_t<T>*>(k), reinterpret_cast<const xdnn_t<T>*>(v),
      reinterpret_cast<xdnn_t<T>*>(k_cache), reinterpret_cast<xdnn_t<T>*>(v_cache), slot_mapping, num_tokens,
      num_blocks, num_heads,
      /*k_head_size=*/head_size, /*v_head_size=*/head_size, block_size);
  KLLM_LOG_DEBUG << "XpuReshapeCachedKV returned: " << ret;
  return ret;
}

template int XpuReshapeCachedKV<float16>(xdnn::Context*, const float16*, const float16*, float16*, float16*, int32_t*,
                                         int64_t, int64_t, int64_t, int64_t, int64_t);
template int XpuReshapeCachedKV<bfloat16>(xdnn::Context*, const bfloat16*, const bfloat16*, bfloat16*, bfloat16*,
                                          int32_t*, int64_t, int64_t, int64_t, int64_t, int64_t);

// fp32 stub: reachable only via DISPATCH_BY_DTYPE_AND_KVTYPE TYPE_FP32 branch (never at runtime).
template <>
int XpuReshapeCachedKV<float>(xdnn::Context*, const float*, const float*, float*, float*, int32_t*, int64_t, int64_t,
                              int64_t, int64_t, int64_t) {
  KLLM_LOG_ERROR << "XpuReshapeCachedKV<float>: fp32 KV cache not supported on Kunlun";
  return -1;
}

// ── Paged Attention ───────────────────────────────────────────────────────────

// Per-T trait for xfa::paged_attention template args.
// fp16: all-fp16 variant (verified working).
// bf16: align with xsglang FA_DECODE<bfloat16> — TL/TGEMM=float (full fp32 accumulator).
template <typename T>
struct PagedAttnTrait {
  using TQ = xdnn_t<T>;
  using TKV = xdnn_t<T>;
  using TO = xdnn_t<T>;
  using TL = xdnn_t<T>;
  using TGEMM0 = xdnn_t<T>;
  using TGEMM1 = xdnn_t<T>;
  using TEW = xdnn_t<T>;
};
template <>
struct PagedAttnTrait<bfloat16> {
  using TQ = xdnn_bf16;
  using TKV = xdnn_bf16;
  using TO = xdnn_bf16;
  using TL = float;
  using TGEMM0 = float;
  using TGEMM1 = float;
  using TEW = float;
};

template <typename T, typename TID>
int XpuPagedAttention(xdnn::Context* ctx, T* output, const T* query, const T* k_cache, const T* v_cache,
                      int64_t num_kv_heads, float scale, const TID* block_table, const TID* context_len,
                      int64_t block_size, int64_t max_context_len, int64_t batch_num, int64_t head_num,
                      int64_t head_dim, int64_t max_num_blocks_per_seq) {
  KLLM_LOG_DEBUG << "XpuPagedAttention: batch=" << batch_num << " heads=" << head_num << " kv_heads=" << num_kv_heads
                 << " head_dim=" << head_dim << " block_size=" << block_size << " max_ctx=" << max_context_len
                 << " max_blocks=" << max_num_blocks_per_seq;

  using Tr = PagedAttnTrait<T>;
  using xfa_t = xdnn_t<T>;
  int ret = xfa::paged_attention<typename Tr::TQ, typename Tr::TKV, typename Tr::TO, typename Tr::TL,
                                 typename Tr::TGEMM0, typename Tr::TGEMM1, typename Tr::TEW, TID, 0>(
      ctx, reinterpret_cast<xfa_t*>(output), reinterpret_cast<const xfa_t*>(query),
      reinterpret_cast<const xfa_t*>(k_cache), reinterpret_cast<const xfa_t*>(v_cache), num_kv_heads, scale,
      block_table, context_len, block_size, max_context_len,
      /*alibi_slopes=*/nullptr, batch_num, head_num, head_dim, max_num_blocks_per_seq);
  KLLM_LOG_DEBUG << "XpuPagedAttention returned: " << ret;
  return ret;
}

template int XpuPagedAttention<float16, int32_t>(xdnn::Context*, float16*, const float16*, const float16*,
                                                 const float16*, int64_t, float, const int32_t*, const int32_t*,
                                                 int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

template int XpuPagedAttention<bfloat16, int32_t>(xdnn::Context*, bfloat16*, const bfloat16*, const bfloat16*,
                                                  const bfloat16*, int64_t, float, const int32_t*, const int32_t*,
                                                  int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

// fp32 stub: DISPATCH_BY_DTYPE_AND_KVTYPE expands TYPE_FP32; never called at runtime.
template <>
int XpuPagedAttention<float, int32_t>(xdnn::Context*, float*, const float*, const float*, const float*, int64_t, float,
                                      const int32_t*, const int32_t*, int64_t, int64_t, int64_t, int64_t, int64_t,
                                      int64_t) {
  KLLM_LOG_ERROR << "XpuPagedAttention<float>: fp32 attention not supported on Kunlun";
  return -1;
}

// ── MLA: dpsk_prefill_rotary_embedding ───────────────────────────────────────

template <typename T, typename TR, typename TID>
int XpuDpskPrefillRotaryEmbedding(xdnn::Context* ctx, const T* kv, const TR* rotary_pos_emb, T* q, T* latent_cache,
                                  T* k, T* v, const TID* positions, int64_t q_head_num, int64_t kv_head_num,
                                  int64_t q_head_dim, int64_t kv_head_dim, int64_t latent_cache_dim, int64_t rot_dim,
                                  int64_t num_tokens) {
  using xfa_t = xdnn_t<T>;
  using xfa_tr = xdnn_t<TR>;
  using xfa_tid = TID;
  xdnn::ctx_guard guard(ctx);
  // TC template param: use same type as TR (cos_sin cache type)
  return infer_ops::dpsk_prefill_rotary_embedding<xfa_t, xfa_tr, xfa_tr, xfa_tid>(
      ctx, reinterpret_cast<const xfa_t*>(kv), reinterpret_cast<const xfa_tr*>(rotary_pos_emb),
      reinterpret_cast<xfa_t*>(q), reinterpret_cast<xfa_t*>(latent_cache), reinterpret_cast<xfa_t*>(k),
      reinterpret_cast<xfa_t*>(v), positions, q_head_num, kv_head_num, q_head_dim, kv_head_dim, latent_cache_dim,
      rot_dim, num_tokens);
}

template int XpuDpskPrefillRotaryEmbedding<float16, float, int64_t>(xdnn::Context*, const float16*, const float*,
                                                                    float16*, float16*, float16*, float16*,
                                                                    const int64_t*, int64_t, int64_t, int64_t, int64_t,
                                                                    int64_t, int64_t, int64_t);
template int XpuDpskPrefillRotaryEmbedding<bfloat16, float, int64_t>(xdnn::Context*, const bfloat16*, const float*,
                                                                     bfloat16*, bfloat16*, bfloat16*, bfloat16*,
                                                                     const int64_t*, int64_t, int64_t, int64_t, int64_t,
                                                                     int64_t, int64_t, int64_t);

// ── MLA: dpsk_decode_rotary_embedding ────────────────────────────────────────

template <typename T, typename TR, typename TID>
int XpuDpskDecodeRotaryEmbedding(xdnn::Context* ctx, T* query, T* key, const TR* rotary_pos_emb, const TID* positions,
                                 int64_t max_seq_len, int64_t rot_dim, int64_t query_stride, int64_t key_stride,
                                 int64_t query_offset, int64_t key_offset, int64_t num_heads, int64_t num_kv_heads,
                                 int64_t num_tokens) {
  using xfa_t = xdnn_t<T>;
  using xfa_tr = xdnn_t<TR>;
  xdnn::ctx_guard guard(ctx);
  // TC = TR, TK = T (same key type as query)
  return infer_ops::dpsk_decode_rotary_embedding<xfa_t, xfa_tr, TID, xfa_tr, xfa_t>(
      ctx, reinterpret_cast<xfa_t*>(query), reinterpret_cast<xfa_t*>(key),
      reinterpret_cast<const xfa_tr*>(rotary_pos_emb), positions, max_seq_len, rot_dim, query_stride, key_stride,
      query_offset, key_offset, num_heads, num_kv_heads, num_tokens);
}

template int XpuDpskDecodeRotaryEmbedding<float16, float, int64_t>(xdnn::Context*, float16*, float16*, const float*,
                                                                   const int64_t*, int64_t, int64_t, int64_t, int64_t,
                                                                   int64_t, int64_t, int64_t, int64_t, int64_t);
template int XpuDpskDecodeRotaryEmbedding<bfloat16, float, int64_t>(xdnn::Context*, bfloat16*, bfloat16*, const float*,
                                                                    const int64_t*, int64_t, int64_t, int64_t, int64_t,
                                                                    int64_t, int64_t, int64_t, int64_t, int64_t);

// ── MLA: KV Cache Write ──────────────────────────────────────────────────────

template <typename T>
int XpuMlaReshapeCachedKV(xdnn::Context* ctx, const T* kv_data, T* kv_cache, int32_t* slot_mapping, int64_t num_tokens,
                          int64_t num_blocks, int64_t kv_cache_dim, int64_t block_size) {
  // MLA stores compressed [latent|k_pe] as a single head: num_heads=1, k_head_size=kv_cache_dim, v=nullptr.
  // Align with xsglang: v_head_size = kv_cache_dim, block_size = 1.
  (void)block_size;
  const xdnn_t<T>* v_null = nullptr;
  xdnn_t<T>* v_cache_null = nullptr;
  return infer_ops::reshape_and_cache<xdnn_t<T>, int32_t>(ctx, reinterpret_cast<const xdnn_t<T>*>(kv_data),
                                                          /*v=*/v_null, reinterpret_cast<xdnn_t<T>*>(kv_cache),
                                                          /*v_cache=*/v_cache_null, slot_mapping, num_tokens,
                                                          num_blocks, /*num_heads=*/1,
                                                          /*k_head_size=*/kv_cache_dim, /*v_head_size=*/kv_cache_dim,
                                                          /*block_size=*/1);
}

template int XpuMlaReshapeCachedKV<float16>(xdnn::Context*, const float16*, float16*, int32_t*, int64_t, int64_t,
                                            int64_t, int64_t);
template int XpuMlaReshapeCachedKV<bfloat16>(xdnn::Context*, const bfloat16*, bfloat16*, int32_t*, int64_t, int64_t,
                                             int64_t, int64_t);

// ── MLA Flash Attention Trait ─────────────────────────────────────────────────
// Encodes per-dtype accumulator configuration for xfa::flash_attention_context_vllm.
// fp16: all-fp16 (TBIAS/TGEMM0/TGEMM1/TEW = xdnn_fp16).
// bf16: fp32 accumulator (TBIAS/TGEMM0/TGEMM1/TEW = float), matching xsglang convention.
template <typename T>
struct FlashMlaAttnTrait {
  using TQ = xdnn_t<T>;
  using TK = xdnn_t<T>;
  using TV = xdnn_t<T>;
  using TBIAS = xdnn_t<T>;
  using TO = xdnn_t<T>;
  using TGEMM0 = xdnn_t<T>;
  using TGEMM1 = xdnn_t<T>;
  using TEW = float;
};
template <>
struct FlashMlaAttnTrait<bfloat16> {
  using TQ = xdnn_bf16;
  using TK = xdnn_bf16;
  using TV = xdnn_bf16;
  using TBIAS = float;
  using TO = xdnn_bf16;
  using TGEMM0 = float;
  using TGEMM1 = float;
  using TEW = float;
};

// ── MLA: Flash Attention (prefill, with vo_head_dim) ─────────────────────────

template <typename T, typename TID>
int XpuFlashMlaAttention(xdnn::Context* ctx, const T* q, const T* k, const T* v, T* output, int64_t batch_num,
                         int64_t max_seq_q, int64_t max_seq_kv, int64_t head_num, int64_t kv_head_num,
                         int64_t qk_head_dim, int64_t v_head_dim, bool is_causal, float softmax_scale, const TID* qlod,
                         int qlod_len, const TID* kvlod, int kvlod_len, int64_t block_size, const TID* qlod_device,
                         const TID* kvlod_device) {
  using Tr = FlashMlaAttnTrait<T>;
  using xfa_t = xdnn_t<T>;

  xdnn::VectorParam<TID> qlod_param = {qlod, qlod_len, const_cast<TID*>(qlod_device)};
  xdnn::VectorParam<TID> kvlod_param = {kvlod, kvlod_len, const_cast<TID*>(kvlod_device)};
  std::vector<int64_t> mask_bias_shape = {};
  std::vector<int64_t> alibi_slopes_shape = {};

  // Compute q_maxptr and k_maxptr
  float* q_maxptr_dev = nullptr;
  void* q_max_buffer = nullptr;
  float* k_maxptr_dev = nullptr;
  void* k_max_buffer = nullptr;
  int64_t total_tokens_q = (qlod != nullptr && qlod_len >= 2) ? qlod[qlod_len - 1] : (batch_num * max_seq_q);
  int64_t q_total_elems = total_tokens_q * head_num * qk_head_dim;
  int64_t total_tokens_kv = (kvlod != nullptr && kvlod_len >= 2) ? kvlod[kvlod_len - 1] : (batch_num * max_seq_kv);
  int64_t k_total_elems = total_tokens_kv * kv_head_num * qk_head_dim;
  size_t maxptr_bytes = static_cast<size_t>(ctx->max_ptr_size()) * sizeof(float);
  if (xpu_malloc(&q_max_buffer, maxptr_bytes) == 0 && q_max_buffer != nullptr) {
    q_maxptr_dev = reinterpret_cast<float*>(q_max_buffer);
    if (xdnn::findmax<xfa_t>(ctx, reinterpret_cast<const xfa_t*>(q), q_maxptr_dev, q_total_elems) != 0) {
      xpu_free(q_max_buffer);
      q_max_buffer = nullptr;
      q_maxptr_dev = nullptr;
    }
  }
  if (xpu_malloc(&k_max_buffer, maxptr_bytes) == 0 && k_max_buffer != nullptr) {
    k_maxptr_dev = reinterpret_cast<float*>(k_max_buffer);
    if (xdnn::findmax<xfa_t>(ctx, reinterpret_cast<const xfa_t*>(k), k_maxptr_dev, k_total_elems) != 0) {
      xpu_free(k_max_buffer);
      k_max_buffer = nullptr;
      k_maxptr_dev = nullptr;
    }
  }

  int ret = xfa::flash_attention_context_vllm<typename Tr::TQ, typename Tr::TK, typename Tr::TV, typename Tr::TBIAS,
                                              typename Tr::TO, typename Tr::TGEMM0, typename Tr::TGEMM1,
                                              typename Tr::TEW, TID, 0>(
      ctx, reinterpret_cast<const xfa_t*>(q), reinterpret_cast<const xfa_t*>(k), reinterpret_cast<const xfa_t*>(v),
      /*mask_bias=*/nullptr, reinterpret_cast<xfa_t*>(output), batch_num, max_seq_q, max_seq_kv, head_num, qk_head_dim,
      kv_head_num,
      /*alibi_slopes=*/nullptr, qlod_param, kvlod_param, mask_bias_shape, alibi_slopes_shape, is_causal,
      /*is_qkv_fusion=*/false,
      /*qkv_layout=*/0x0010,  // ATTN_BLHD
      /*vsl_type=*/0x10,      // ATTN_LVSL
      q_maxptr_dev, k_maxptr_dev, nullptr, nullptr,
      /*vo_head_dim=*/v_head_dim,
      /*k_cache=*/nullptr, /*v_cache=*/nullptr, nullptr, nullptr,
      /*block_size=*/block_size, /*max_blocks_per_seq=*/-1, /*prefill_len=*/-1, /*block_tables=*/nullptr, softmax_scale,
      /*swa_left_size=*/-1, /*swa_right_size=*/-1,
      /*softmax_lse=*/nullptr, /*unpadded_lse=*/false, /*attn_sink=*/nullptr);

  if (q_max_buffer != nullptr) xpu_free(q_max_buffer);
  if (k_max_buffer != nullptr) xpu_free(k_max_buffer);
  return ret;
}

template int XpuFlashMlaAttention<float16, int32_t>(xdnn::Context*, const float16*, const float16*, const float16*,
                                                    float16*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                                    int64_t, bool, float, const int32_t*, int, const int32_t*, int,
                                                    int64_t, const int32_t*, const int32_t*);
template int XpuFlashMlaAttention<bfloat16, int32_t>(xdnn::Context*, const bfloat16*, const bfloat16*, const bfloat16*,
                                                     bfloat16*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                                     int64_t, bool, float, const int32_t*, int, const int32_t*, int,
                                                     int64_t, const int32_t*, const int32_t*);

// ── MLA: Paged Attention (decode on compressed cache) ────────────────────────

// ── MLA Paged Attention Trait ────────────────────────────────────────────────
// Differs from PagedAttnTrait: TEW=float for fp16 (matches xsglang speculative_multi_latent_attention).
template <typename T>
struct MlaPagedAttnTrait {
  using TQ = xdnn_t<T>;
  using TKV = xdnn_t<T>;
  using TO = xdnn_t<T>;
  using TGEMM0 = xdnn_t<T>;
  using TGEMM1 = xdnn_t<T>;
  using TEW = float;  // xfa MLA kernel uses float for element-wise accumulation
};
template <>
struct MlaPagedAttnTrait<bfloat16> {
  using TQ = xdnn_bf16;
  using TKV = xdnn_bf16;
  using TO = xdnn_bf16;
  using TGEMM0 = float;
  using TGEMM1 = float;
  using TEW = float;
};

template <typename T, typename TID>
int XpuPagedMlaAttention(xdnn::Context* ctx, T* output, const T* query, const T* kv_cache, const TID* block_table,
                         const TID* context_len_device, const TID* context_len_host, int64_t batch_num,
                         int64_t q_seq_len, int64_t num_heads, int64_t kv_lora_rank, int64_t qk_rope_head_dim,
                         float scale, int64_t block_size, int64_t max_context_len, int64_t max_num_blocks_per_seq) {
  using Tr = MlaPagedAttnTrait<T>;
  using xfa_t = xdnn_t<T>;

  // VectorParam requires both host and device pointers for context_len
  xdnn::VectorParam<TID> seq_lengths = {const_cast<TID*>(context_len_host), batch_num,
                                        const_cast<TID*>(context_len_device)};

  // valid_batch: must provide [0, 1, ..., batch_num-1] on host.
  // The xfa kernel uses this to index which batch elements to process.
  // Passing {nullptr, 0, nullptr} is NOT acceptable — the kernel returns error.
  // VectorParam::to_xpu() will auto-copy host → device internally if xpu==nullptr.
  //
  // Held in a thread_local buffer rather than a stack vector for two reasons:
  //   1. decode calls this once per layer per step, so a fresh vector means an O(num_layers)
  //      malloc/free burst on the critical path for a payload of a few dozen bytes;
  //   2. the host pointer below is handed to the kernel, which copies it out during to_xpu().
  //      A stack vector dies the moment this function returns, so its lifetime is strictly
  //      shorter than the buffer's; a thread_local one outlives the call.
  // Each rank runs on its own thread, so there is no cross-thread sharing. The contents are a
  // pure prefix sequence, so growing only ever appends and never rewrites live entries.
  thread_local std::vector<TID> valid_batch_host;
  if (static_cast<int64_t>(valid_batch_host.size()) < batch_num) {
    const int64_t old_size = static_cast<int64_t>(valid_batch_host.size());
    valid_batch_host.resize(batch_num);
    for (int64_t i = old_size; i < batch_num; ++i) {
      valid_batch_host[i] = static_cast<TID>(i);
    }
  }
  xdnn::VectorParam<TID> valid_batch = {valid_batch_host.data(), batch_num,
                                        nullptr};  // kernel's to_xpu() will auto-alloc+copy

  int ret = xfa::speculative_multi_latent_attention<typename Tr::TQ, typename Tr::TKV, typename Tr::TO,
                                                    typename Tr::TGEMM0, typename Tr::TGEMM1, typename Tr::TEW, TID, 0>(
      ctx, reinterpret_cast<typename Tr::TO*>(output), reinterpret_cast<const typename Tr::TQ*>(query),
      /*kv=*/nullptr, reinterpret_cast<const typename Tr::TKV*>(kv_cache), block_table, seq_lengths, valid_batch,
      batch_num, q_seq_len, max_context_len, num_heads, kv_lora_rank, qk_rope_head_dim,
      /*attn_mask=*/nullptr, scale, block_size, max_num_blocks_per_seq,
      /*max_window_size=*/-1,
      /*q_maxptr=*/nullptr, /*kv_cache_maxptr=*/nullptr, /*o_maxptr=*/nullptr);

  return ret;
}

template int XpuPagedMlaAttention<float16, int32_t>(xdnn::Context*, float16*, const float16*, const float16*,
                                                    const int32_t*, const int32_t*, const int32_t*, int64_t, int64_t,
                                                    int64_t, int64_t, int64_t, float, int64_t, int64_t, int64_t);
template int XpuPagedMlaAttention<bfloat16, int32_t>(xdnn::Context*, bfloat16*, const bfloat16*, const bfloat16*,
                                                     const int32_t*, const int32_t*, const int32_t*, int64_t, int64_t,
                                                     int64_t, int64_t, int64_t, float, int64_t, int64_t, int64_t);

// ── RMSNorm Split Col ────────────────────────────────────────────────────────

template <typename T, typename TW>
int XpuRmsnormSplitCol(xdnn::Context* ctx, const T* input, const TW* weight0, const TW* weight1, T* output_q,
                       T* output_kv, T* output_latent, int64_t rows, int64_t q_dim, int64_t kv_dim, int64_t rot_dim,
                       float eps) {
  int64_t n = q_dim + kv_dim + rot_dim;
  int64_t latent_dim = kv_dim + rot_dim;
  return infer_ops::rmsnorm_split_col<xdnn_t<T>, xdnn_t<TW>>(
      ctx, xcast(input), xcast(weight0), xcast(weight1),
      /*m=*/rows, /*n=*/n, /*weight0_dim=*/q_dim, /*weight1_dim=*/kv_dim,
      /*q=*/xcast_m(output_q), /*k=*/xcast_m(output_kv), /*v=*/xcast_m(output_latent),
      /*q_dim=*/q_dim, /*k_dim=*/kv_dim, /*v_dim=*/latent_dim, eps, /*force_sdnn=*/false);
}

// TW=float variants: retained for fp32 norm weights (e.g. unit tests / legacy paths).
template int XpuRmsnormSplitCol<float16, float>(xdnn::Context*, const float16*, const float*, const float*, float16*,
                                                float16*, float16*, int64_t, int64_t, int64_t, int64_t, float);
template int XpuRmsnormSplitCol<bfloat16, float>(xdnn::Context*, const bfloat16*, const float*, const float*, bfloat16*,
                                                 bfloat16*, bfloat16*, int64_t, int64_t, int64_t, int64_t, float);
// TW=T (16-bit) variants: used by KunlunRmsnormSplitColLayer now that norm weights keep
// their native bf16/fp16 dtype. Backed by infer_ops::rmsnorm_split_col<{float16,bfloat16}, same>.
template int XpuRmsnormSplitCol<float16, float16>(xdnn::Context*, const float16*, const float16*, const float16*,
                                                  float16*, float16*, float16*, int64_t, int64_t, int64_t, int64_t,
                                                  float);
template int XpuRmsnormSplitCol<bfloat16, bfloat16>(xdnn::Context*, const bfloat16*, const bfloat16*, const bfloat16*,
                                                    bfloat16*, bfloat16*, bfloat16*, int64_t, int64_t, int64_t, int64_t,
                                                    float);

// ── MLA Prefix Cache: Gather from paged cache ───────────────────────────────

template <typename T, typename TID>
int XpuGatherMlaCache(xdnn::Context* ctx, const T* kv_cache, T* kv_c_out, T* k_pe_out, int64_t kv_lora_rank,
                      int64_t qk_rope_dim, const TID* seq_lod, int seq_lod_len, const TID* seq_lod_device,
                      const TID* block_table, int64_t block_size, int64_t max_num_blocks_per_seq) {
  using xfa_t = xdnn_t<T>;
  xdnn::VectorParam<TID> seq_lod_param = {const_cast<TID*>(seq_lod), seq_lod_len, const_cast<TID*>(seq_lod_device)};

  return infer_ops::gather_cache<xfa_t, TID>(ctx, reinterpret_cast<const xfa_t*>(kv_cache),
                                             reinterpret_cast<xfa_t*>(kv_c_out), reinterpret_cast<xfa_t*>(k_pe_out),
                                             kv_lora_rank, qk_rope_dim, seq_lod_param, block_table, block_size,
                                             max_num_blocks_per_seq);
}

template int XpuGatherMlaCache<float16, int32_t>(xdnn::Context*, const float16*, float16*, float16*, int64_t, int64_t,
                                                 const int32_t*, int, const int32_t*, const int32_t*, int64_t, int64_t);
template int XpuGatherMlaCache<bfloat16, int32_t>(xdnn::Context*, const bfloat16*, bfloat16*, bfloat16*, int64_t,
                                                  int64_t, const int32_t*, int, const int32_t*, const int32_t*, int64_t,
                                                  int64_t);

// ── MLA Prefix Cache: Flash Attention with LSE ──────────────────────────────

template <typename T, typename TID>
int XpuFlashMlaAttentionWithLse(xdnn::Context* ctx, const T* q, const T* k, const T* v, T* output, float* softmax_lse,
                                bool unpadded_lse, int64_t batch_num, int64_t max_seq_q, int64_t max_seq_kv,
                                int64_t head_num, int64_t kv_head_num, int64_t qk_head_dim, int64_t v_head_dim,
                                bool is_causal, float softmax_scale, const TID* qlod, int qlod_len, const TID* kvlod,
                                int kvlod_len, int64_t block_size, const TID* qlod_device, const TID* kvlod_device) {
  using Tr = FlashMlaAttnTrait<T>;
  using xfa_t = xdnn_t<T>;

  xdnn::VectorParam<TID> qlod_param = {qlod, qlod_len, const_cast<TID*>(qlod_device)};
  xdnn::VectorParam<TID> kvlod_param = {kvlod, kvlod_len, const_cast<TID*>(kvlod_device)};
  std::vector<int64_t> mask_bias_shape = {};
  std::vector<int64_t> alibi_slopes_shape = {};

  // Compute q_maxptr and k_maxptr
  float* q_maxptr_dev = nullptr;
  void* q_max_buffer = nullptr;
  float* k_maxptr_dev = nullptr;
  void* k_max_buffer = nullptr;
  int64_t total_tokens_q = (qlod != nullptr && qlod_len >= 2) ? qlod[qlod_len - 1] : (batch_num * max_seq_q);
  int64_t q_total_elems = total_tokens_q * head_num * qk_head_dim;
  int64_t total_tokens_kv = (kvlod != nullptr && kvlod_len >= 2) ? kvlod[kvlod_len - 1] : (batch_num * max_seq_kv);
  int64_t k_total_elems = total_tokens_kv * kv_head_num * qk_head_dim;
  size_t maxptr_bytes = static_cast<size_t>(ctx->max_ptr_size()) * sizeof(float);
  if (xpu_malloc(&q_max_buffer, maxptr_bytes) == 0 && q_max_buffer != nullptr) {
    q_maxptr_dev = reinterpret_cast<float*>(q_max_buffer);
    if (xdnn::findmax<xfa_t>(ctx, reinterpret_cast<const xfa_t*>(q), q_maxptr_dev, q_total_elems) != 0) {
      xpu_free(q_max_buffer);
      q_max_buffer = nullptr;
      q_maxptr_dev = nullptr;
    }
  }
  if (xpu_malloc(&k_max_buffer, maxptr_bytes) == 0 && k_max_buffer != nullptr) {
    k_maxptr_dev = reinterpret_cast<float*>(k_max_buffer);
    if (xdnn::findmax<xfa_t>(ctx, reinterpret_cast<const xfa_t*>(k), k_maxptr_dev, k_total_elems) != 0) {
      xpu_free(k_max_buffer);
      k_max_buffer = nullptr;
      k_maxptr_dev = nullptr;
    }
  }

  int ret = xfa::flash_attention_context_vllm<typename Tr::TQ, typename Tr::TK, typename Tr::TV, typename Tr::TBIAS,
                                              typename Tr::TO, typename Tr::TGEMM0, typename Tr::TGEMM1,
                                              typename Tr::TEW, TID, 0>(
      ctx, reinterpret_cast<const xfa_t*>(q), reinterpret_cast<const xfa_t*>(k), reinterpret_cast<const xfa_t*>(v),
      /*mask_bias=*/nullptr, reinterpret_cast<xfa_t*>(output), batch_num, max_seq_q, max_seq_kv, head_num, qk_head_dim,
      kv_head_num,
      /*alibi_slopes=*/nullptr, qlod_param, kvlod_param, mask_bias_shape, alibi_slopes_shape, is_causal,
      /*is_qkv_fusion=*/false,
      /*qkv_layout=*/0x0010,  // ATTN_BLHD
      /*vsl_type=*/0x10,      // ATTN_LVSL
      q_maxptr_dev, k_maxptr_dev, nullptr, nullptr,
      /*vo_head_dim=*/v_head_dim,
      /*k_cache=*/nullptr, /*v_cache=*/nullptr, nullptr, nullptr,
      /*block_size=*/block_size, /*max_blocks_per_seq=*/-1, /*prefill_len=*/-1, /*block_tables=*/nullptr, softmax_scale,
      /*swa_left_size=*/-1, /*swa_right_size=*/-1,
      /*softmax_lse=*/softmax_lse, /*unpadded_lse=*/unpadded_lse, /*attn_sink=*/nullptr);

  if (q_max_buffer != nullptr) xpu_free(q_max_buffer);
  if (k_max_buffer != nullptr) xpu_free(k_max_buffer);
  return ret;
}

template int XpuFlashMlaAttentionWithLse<float16, int32_t>(xdnn::Context*, const float16*, const float16*,
                                                           const float16*, float16*, float*, bool, int64_t, int64_t,
                                                           int64_t, int64_t, int64_t, int64_t, int64_t, bool, float,
                                                           const int32_t*, int, const int32_t*, int, int64_t,
                                                           const int32_t*, const int32_t*);
template int XpuFlashMlaAttentionWithLse<bfloat16, int32_t>(xdnn::Context*, const bfloat16*, const bfloat16*,
                                                            const bfloat16*, bfloat16*, float*, bool, int64_t, int64_t,
                                                            int64_t, int64_t, int64_t, int64_t, int64_t, bool, float,
                                                            const int32_t*, int, const int32_t*, int, int64_t,
                                                            const int32_t*, const int32_t*);

// ── MLA Prefix Cache: Attention Merge State ─────────────────────────────────

template <typename T>
int XpuAttentionMergeState(xdnn::Context* ctx, const T* v_a, const float* s_a, const T* v_b, const float* s_b,
                           T* v_merged, float* s_merged, int64_t num_tokens, int64_t num_heads, int64_t head_dim) {
  using xfa_t = xdnn_t<T>;
  return xfa::attention_merge_stage<xfa_t, float>(
      ctx, reinterpret_cast<const xfa_t*>(v_a), s_a, reinterpret_cast<const xfa_t*>(v_b), s_b,
      reinterpret_cast<xfa_t*>(v_merged), s_merged, num_tokens, num_heads, head_dim);
}

template int XpuAttentionMergeState<float16>(xdnn::Context*, const float16*, const float*, const float16*, const float*,
                                             float16*, float*, int64_t, int64_t, int64_t);
template int XpuAttentionMergeState<bfloat16>(xdnn::Context*, const bfloat16*, const float*, const bfloat16*,
                                              const float*, bfloat16*, float*, int64_t, int64_t, int64_t);

// ── MLA Prefix Cache: K_PE Expand ──────────────────────────────────────────

template <typename T>
int XpuKpeExpand(xdnn::Context* ctx, const T* k_pe, T* v_k, int64_t tokens, int64_t head_num, int64_t v_head_dim,
                 int64_t k_nope_dim, int64_t k_rope_dim) {
  using xfa_t = xdnn_t<T>;
  return infer_ops::k_pe_expand<xfa_t>(ctx, reinterpret_cast<const xfa_t*>(k_pe), reinterpret_cast<xfa_t*>(v_k), tokens,
                                       head_num, v_head_dim, k_nope_dim, k_rope_dim);
}

template int XpuKpeExpand<float16>(xdnn::Context*, const float16*, float16*, int64_t, int64_t, int64_t, int64_t,
                                   int64_t);
template int XpuKpeExpand<bfloat16>(xdnn::Context*, const bfloat16*, bfloat16*, int64_t, int64_t, int64_t, int64_t,
                                    int64_t);

// ── MLA: fp32 stubs (never called at runtime, required by DISPATCH_BY_DTYPE_AND_KVTYPE) ──

template <>
int XpuDpskPrefillRotaryEmbedding<float, float, int64_t>(xdnn::Context*, const float*, const float*, float*, float*,
                                                         float*, float*, const int64_t*, int64_t, int64_t, int64_t,
                                                         int64_t, int64_t, int64_t, int64_t) {
  KLLM_LOG_ERROR << "XpuDpskPrefillRotaryEmbedding<float>: fp32 not supported on Kunlun";
  return -1;
}

template <>
int XpuDpskDecodeRotaryEmbedding<float, float, int64_t>(xdnn::Context*, float*, float*, const float*, const int64_t*,
                                                        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                                        int64_t, int64_t) {
  KLLM_LOG_ERROR << "XpuDpskDecodeRotaryEmbedding<float>: fp32 not supported on Kunlun";
  return -1;
}

template <>
int XpuMlaReshapeCachedKV<float>(xdnn::Context*, const float*, float*, int32_t*, int64_t, int64_t, int64_t, int64_t) {
  KLLM_LOG_ERROR << "XpuMlaReshapeCachedKV<float>: fp32 not supported on Kunlun";
  return -1;
}

template <>
int XpuFlashMlaAttention<float, int32_t>(xdnn::Context*, const float*, const float*, const float*, float*, int64_t,
                                         int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, float,
                                         const int32_t*, int, const int32_t*, int, int64_t, const int32_t*,
                                         const int32_t*) {
  KLLM_LOG_ERROR << "XpuFlashMlaAttention<float>: fp32 not supported on Kunlun";
  return -1;
}

template <>
int XpuPagedMlaAttention<float, int32_t>(xdnn::Context*, float*, const float*, const float*, const int32_t*,
                                         const int32_t*, const int32_t*, int64_t, int64_t, int64_t, int64_t, int64_t,
                                         float, int64_t, int64_t, int64_t) {
  KLLM_LOG_ERROR << "XpuPagedMlaAttention<float>: fp32 not supported on Kunlun";
  return -1;
}

template <>
int XpuRmsnormSplitCol<float, float>(xdnn::Context*, const float*, const float*, const float*, float*, float*, float*,
                                     int64_t, int64_t, int64_t, int64_t, float) {
  KLLM_LOG_ERROR << "XpuRmsnormSplitCol<float>: fp32 not supported on Kunlun";
  return -1;
}

template <>
int XpuGatherMlaCache<float, int32_t>(xdnn::Context*, const float*, float*, float*, int64_t, int64_t, const int32_t*,
                                      int, const int32_t*, const int32_t*, int64_t, int64_t) {
  KLLM_LOG_ERROR << "XpuGatherMlaCache<float>: fp32 not supported on Kunlun";
  return -1;
}

template <>
int XpuFlashMlaAttentionWithLse<float, int32_t>(xdnn::Context*, const float*, const float*, const float*, float*,
                                                float*, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                                int64_t, bool, float, const int32_t*, int, const int32_t*, int, int64_t,
                                                const int32_t*, const int32_t*) {
  KLLM_LOG_ERROR << "XpuFlashMlaAttentionWithLse<float>: fp32 not supported on Kunlun";
  return -1;
}

template <>
int XpuAttentionMergeState<float>(xdnn::Context*, const float*, const float*, const float*, const float*, float*,
                                  float*, int64_t, int64_t, int64_t) {
  KLLM_LOG_ERROR << "XpuAttentionMergeState<float>: fp32 not supported on Kunlun";
  return -1;
}

template <>
int XpuKpeExpand<float>(xdnn::Context*, const float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t) {
  KLLM_LOG_ERROR << "XpuKpeExpand<float>: fp32 not supported on Kunlun";
  return -1;
}

}  // namespace ksana_llm
