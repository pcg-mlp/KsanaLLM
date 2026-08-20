/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

// XPU 基础头单独成块，确保排在 xfa/infer_ops 之前（.clang-format 里 IncludeBlocks: Preserve，
// 块内排序、块间不重排）。flash_api.h 来自 XMLIR SDK 且未验证自包含，不要与本块合并。
#include <xpu/runtime.h>
#include <xpu/xdnn.h>

#include <flash_api.h>       // xfa::flash_attention_context_vllm, xfa::paged_attention
#include <infer_ops.h>       // infer_ops::reshape_and_cache
#include <infer_ops_rope.h>  // infer_ops::rotary_embedding_neox

#include <cstdint>

#include "kernels/kunlun/basic_kernel_wrapper.h"

namespace ksana_llm {

// Template naming convention: see KUNLUN_TEMPLATE_NAMING.md for the shared glossary
// (T, TW, TID, ...). This file additionally uses:
//   TCACHE — KV-cache storage dtype (paged/flash attention)
//   TR     — rotary_pos_emb dtype (RoPE)

// rotary position embedding (neox style): in-place modify query and key
// positions [num_tokens], query [num_tokens, num_heads * head_size]
// key [num_tokens, num_kv_heads * head_size], cos_sin_cache [max_pos, rot_dim]
template <typename T>
int XpuRotaryEmbeddingNeox(xdnn::Context* ctx, const int64_t* positions, T* query, T* key, const T* cos_sin_cache,
                           int rot_dim, int num_heads, int num_kv_heads, int head_size, int num_tokens);

// flash attention (prefill/context): pass qlod/kvlod for prefill, block_table for decode
template <typename T, typename TCACHE, typename TID = int32_t>
int XpuFlashAttentionContext(xdnn::Context* ctx, const T* q, const T* k, const T* v, const TCACHE* k_cache,
                             const TCACHE* v_cache, T* output, int64_t batch_num, int64_t max_seq_q, int64_t max_seq_kv,
                             int64_t max_seq_cache, int64_t head_num, int64_t kv_head_num, int64_t head_dim,
                             bool is_causal, float softmax_scale, const TID* qlod, int qlod_len, const TID* kvlod,
                             int kvlod_len, const TID* cachelod, int cachelod_len, const TID* block_table,
                             int64_t block_size, int64_t max_blocks_per_seq);

// KV cache scatter via slot_mapping
template <typename T>
int XpuReshapeCachedKV(xdnn::Context* ctx, const T* k, const T* v, T* k_cache, T* v_cache, int32_t* slot_mapping,
                       int64_t num_tokens, int64_t num_blocks, int64_t num_heads, int64_t head_size,
                       int64_t block_size);

// paged attention (decode): Q vs paged KV cache
template <typename T, typename TID = int32_t>
int XpuPagedAttention(xdnn::Context* ctx, T* output, const T* query, const T* k_cache, const T* v_cache,
                      int64_t num_kv_heads, float scale, const TID* block_table, const TID* context_len,
                      int64_t block_size, int64_t max_context_len, int64_t batch_num, int64_t head_num,
                      int64_t head_dim, int64_t max_num_blocks_per_seq);

// Fused split(QKV) + RMSNorm(Q,K) + RoPE(Q,K) in one kernel launch.
// qkv: [num_tokens, (head_num + 2*kv_head_num) * head_dim]  (packed input)
// rotary_pos_emb: [2, 1, max_seqlen, rotary_dim]  (neox cache: cos-plane then sin-plane, doubled)
// q/k_norm_weight: [head_dim] in type T
// positions: [num_tokens]
// q_out: [num_tokens, head_num * head_dim], k_out/v_out: [num_tokens, kv_head_num * head_dim]
template <typename T>
int XpuSplitNormRopeNeox(xdnn::Context* ctx, const T* qkv, const T* rotary_pos_emb, const T* q_norm_weight,
                         const T* k_norm_weight, const int64_t* positions, T* q_out, T* k_out, T* v_out,
                         int64_t num_tokens, int64_t max_seqlen, int64_t head_num, int64_t kv_head_num,
                         int64_t head_dim, int64_t rotary_dim, float eps);

// Fused split(QKV) + RoPE(Q,K) in a single kernel launch, for models WITHOUT QK-Norm.
// This is the no-norm counterpart of XpuSplitNormRopeNeox: it replaces the
// 3x copy2d + 1x rotary_embedding_neox sequence with one kernel.
//
// qkv: [num_tokens, (head_num + 2*kv_head_num) * head_dim]  (packed input)
// cos_sin_cache: [max_seqlen, rotary_dim] — the SAME buffer/layout consumed by
//                XpuRotaryEmbeddingNeox (i.e. cos_sin_cache_ptr_), NOT the
//                doubled neox cache that XpuSplitNormRopeNeox expects.
// positions: [num_tokens]
// q_out: [num_tokens, head_num * head_dim], k_out/v_out: [num_tokens, kv_head_num * head_dim]
template <typename T>
int XpuSplitRopeNeox(xdnn::Context* ctx, const T* qkv, const T* cos_sin_cache, const int64_t* positions, T* q_out,
                     T* k_out, T* v_out, int64_t num_tokens, int64_t max_seqlen, int64_t head_num, int64_t kv_head_num,
                     int64_t head_dim, int64_t rotary_dim);

// ── MLA Kernels ──────────────────────────────────────────────────────────────

// dpsk_prefill_rotary_embedding: fused q_rope + kv_split(k_nope/v) + latent_cache rope + k_pe broadcast
// kv: [num_tokens, kv_head_num, kv_head_dim * 2] (kv_head_dim = qk_nope_head_dim = v_head_dim for MLA)
// q: [num_tokens, q_head_num, q_head_dim + rot_dim]  (in/out, rope applied to last rot_dim)
// latent_cache: [num_tokens, 1, latent_cache_dim + rot_dim] (in/out, rope applied to last rot_dim)
// k: [num_tokens, kv_head_num, kv_head_dim + rot_dim] (output, k_nope from kv + roped k_pe broadcast)
// v: [num_tokens, kv_head_num, kv_head_dim] (output, split from kv)
template <typename T, typename TR, typename TID = int64_t>
int XpuDpskPrefillRotaryEmbedding(xdnn::Context* ctx, const T* kv, const TR* rotary_pos_emb, T* q, T* latent_cache,
                                  T* k, T* v, const TID* positions, int64_t q_head_num, int64_t kv_head_num,
                                  int64_t q_head_dim, int64_t kv_head_dim, int64_t latent_cache_dim, int64_t rot_dim,
                                  int64_t num_tokens);

// dpsk_decode_rotary_embedding: in-place rope on query and key's last rot_dim dimensions
// query: [num_tokens, num_heads, query_stride] rope in [..., query_offset:query_offset+rot_dim]
// key: [num_tokens, num_kv_heads, key_stride] rope in [..., key_offset:key_offset+rot_dim]
template <typename T, typename TR, typename TID = int64_t>
int XpuDpskDecodeRotaryEmbedding(xdnn::Context* ctx, T* query, T* key, const TR* rotary_pos_emb, const TID* positions,
                                 int64_t max_seq_len, int64_t rot_dim, int64_t query_stride, int64_t key_stride,
                                 int64_t query_offset, int64_t key_offset, int64_t num_heads, int64_t num_kv_heads,
                                 int64_t num_tokens);

// MLA KV cache write: store [latent|k_pe] into paged blocks (num_heads=1, v=nullptr)
template <typename T>
int XpuMlaReshapeCachedKV(xdnn::Context* ctx, const T* kv_data, T* kv_cache, int32_t* slot_mapping, int64_t num_tokens,
                          int64_t num_blocks, int64_t kv_cache_dim, int64_t block_size);

// MLA Prefill Flash Attention: supports vo_head_dim (V output dim ≠ K dim)
template <typename T, typename TID = int32_t>
int XpuFlashMlaAttention(xdnn::Context* ctx, const T* q, const T* k, const T* v, T* output, int64_t batch_num,
                         int64_t max_seq_q, int64_t max_seq_kv, int64_t head_num, int64_t kv_head_num,
                         int64_t qk_head_dim, int64_t v_head_dim, bool is_causal, float softmax_scale, const TID* qlod,
                         int qlod_len, const TID* kvlod, int kvlod_len, int64_t block_size,
                         const TID* qlod_device = nullptr, const TID* kvlod_device = nullptr);

// MLA Decode Paged Attention: decode on compressed [latent|k_pe] cache
// query: [batch_num * q_seq_len, num_heads, kv_lora_rank + qk_rope_head_dim]
// kv_cache: paged blocks of [block_size, 1, kv_lora_rank + qk_rope_head_dim]
template <typename T, typename TID = int32_t>
int XpuPagedMlaAttention(xdnn::Context* ctx, T* output, const T* query, const T* kv_cache, const TID* block_table,
                         const TID* context_len_device, const TID* context_len_host, int64_t batch_num,
                         int64_t q_seq_len, int64_t num_heads, int64_t kv_lora_rank, int64_t qk_rope_head_dim,
                         float scale, int64_t block_size, int64_t max_context_len, int64_t max_num_blocks_per_seq);

// rmsnorm_split_col: fused RMSNorm + column split (wraps infer_ops::rmsnorm_split_col)
// input: [rows, q_dim + kv_dim + rot_dim]
// output_q: [rows, q_dim] — rmsnorm with weight0 (q_a_layernorm)
// output_kv: [rows, kv_dim] — rmsnorm with weight1 (kv_a_layernorm)
// output_latent: [rows, kv_dim + rot_dim] — rmsnorm(kv) concat raw_rope
template <typename T, typename TW>
int XpuRmsnormSplitCol(xdnn::Context* ctx, const T* input, const TW* weight0, const TW* weight1, T* output_q,
                       T* output_kv, T* output_latent, int64_t rows, int64_t q_dim, int64_t kv_dim, int64_t rot_dim,
                       float eps);

// ── MLA Prefix Cache Kernels ─────────────────────────────────────────────────

// Gather MLA compressed KV from paged cache: extracts kv_c and k_pe from paged blocks
// kv_cache: [NUM_BLOCKS, BLOCK_SIZE, kv_lora_rank + qk_rope_dim] (combined latent cache)
// kv_c_out: [sum_seq_length, kv_lora_rank] (gathered kv_a_normed)
// k_pe_out: [sum_seq_length, qk_rope_dim] (gathered k_pe)
// Addressing: src = kv_cache + (block_table[b*max_num_blocks_per_seq + i]*block_size + t) * cache_dim
// seq_lod / seq_lod_len: HOST CSR offsets, len = batch+1; seq b gathers seq_lod[b+1]-seq_lod[b] tokens
//   from token 0, i.e. reads ceil(len/block_size) entries from row b (trailing entries may be -1).
// seq_lod_device: optional DEVICE copy of seq_lod; nullptr lets the wrapper copy it internally.
// block_table: DEVICE [batch, max_num_blocks_per_seq]; values are RAW block ids relative to the passed
//   kv_cache pointer — layer offset must be in the POINTER (kv_cache_layer_ptr), NOT in the ids.
//   Do not pass slot_mapping-style ids (blk + layer_idx) here, that double-counts the layer stride.
// block_size: tokens per block; must equal block_token_num_ (same granularity as block_table/BLOCK_SIZE).
template <typename T, typename TID = int32_t>
int XpuGatherMlaCache(xdnn::Context* ctx, const T* kv_cache, T* kv_c_out, T* k_pe_out, int64_t kv_lora_rank,
                      int64_t qk_rope_dim, const TID* seq_lod, int seq_lod_len, const TID* seq_lod_device,
                      const TID* block_table, int64_t block_size, int64_t max_num_blocks_per_seq);

// Flash MLA Attention with LSE output: same as XpuFlashMlaAttention but returns softmax_lse
template <typename T, typename TID = int32_t>
int XpuFlashMlaAttentionWithLse(xdnn::Context* ctx, const T* q, const T* k, const T* v, T* output, float* softmax_lse,
                                bool unpadded_lse, int64_t batch_num, int64_t max_seq_q, int64_t max_seq_kv,
                                int64_t head_num, int64_t kv_head_num, int64_t qk_head_dim, int64_t v_head_dim,
                                bool is_causal, float softmax_scale, const TID* qlod, int qlod_len, const TID* kvlod,
                                int kvlod_len, int64_t block_size, const TID* qlod_device = nullptr,
                                const TID* kvlod_device = nullptr);

// Attention merge: combine two attention outputs using their LSEs
// v_a, v_b: [num_tokens, num_heads, head_dim]  (attention outputs)
// s_a, s_b: [num_tokens, num_heads] (softmax LSEs, unpadded layout)
// v_merged: [num_tokens, num_heads, head_dim] (merged output)
// s_merged: [num_tokens, num_heads] or nullptr (merged LSE, optional)
template <typename T>
int XpuAttentionMergeState(xdnn::Context* ctx, const T* v_a, const float* s_a, const T* v_b, const float* s_b,
                           T* v_merged, float* s_merged, int64_t num_tokens, int64_t num_heads, int64_t head_dim);

// K_PE expand: broadcast k_pe from [tokens, 1, rope_dim] to [tokens, heads, rope_dim] and
// write into output at the correct position within [tokens, heads, v_head_dim + k_nope_dim + k_rope_dim]
// This wraps infer_ops::k_pe_expand
template <typename T>
int XpuKpeExpand(xdnn::Context* ctx, const T* k_pe, T* v_k, int64_t tokens, int64_t head_num, int64_t v_head_dim,
                 int64_t k_nope_dim, int64_t k_rope_dim);

}  // namespace ksana_llm
