/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/utils/quant_type.h"

namespace llm_kernels {
namespace nvidia {

/**
 * @brief Fused Q per-head RMSNorm + RoPE + KV RoPE + KV FP8 paged-cache insert + Q expand/quant.
 *
 * Combines per-head (no-weight) RMSNorm on Q, GPT-J style RoPE on both Q and KV,
 * FP8-E4M3 quantization of KV with paged cache insertion (SWA), and Q output
 * controlled by KV_DTYPE template parameter:
 *   - kFp8DsMla (default): expand Q from [N, num_heads_q, 512] to [N, 64, 512] bf16 (FlashMLA).
 *   - kFp8E4M3: quantize Q to FP8 E4M3, writing dense fp8 to q_out.
 *
 * Dimensions:
 *   head_dim         = 512
 *   qk_nope_head_dim = 448
 *   qk_rope_head_dim = 64
 *   rotary_dim       = 64 (= qk_rope_head_dim)
 *
 * @param q_in           [num_tokens, num_heads_q * 512]  bf16, input Q (from wq_b).
 * @param kv_in          [num_tokens, 512]                bf16, input KV (normed kv_a)
 * @param q_out          [num_tokens, 64 * 512]           Output Q buffer:
 *                       - kFp8DsMla: bf16 expanded Q with stride=64*512
 *                       - kFp8E4M3: fp8_e4m3 dense Q with stride=64*512
 * @param swa_kv_cache   Paged KV cache byte array (FP8_DS_MLA format)
 * @param positions      [num_tokens]                     int64, position ids for cos/sin lookup
 * @param cos_sin_cache  [max_pos, 64]                    bf16, layout [cos(32), sin(32)] per position
 * @param raw_out_loc    [num_tokens]                     int32, slot indices for SWA cache
 *                       (page = index >> page_bits, offset = index & page_mask)
 *                       -1 means skip store for that token.
 * @param eps            RMSNorm epsilon (e.g. 1e-6)
 * @param num_tokens     Total number of tokens (N)
 * @param num_heads_q    Number of Q heads (must be <= 64)
 * @param page_stride    Stride in bytes between pages in swa_kv_cache
 * @param page_bits      log2(page_size): 1, 6, or 8
 * @param stream         CUDA stream
 */
template <llm_kernels::utils::KVCacheType KV_DTYPE = llm_kernels::utils::KVCacheType::kFp8DsMla>
void InvokeFusedQNormRopeKVInsert(const __nv_bfloat16* q_in, const __nv_bfloat16* kv_in, void* q_out,
                                  void* swa_kv_cache, const int64_t* positions, const __nv_bfloat16* cos_sin_cache,
                                  const int32_t* raw_out_loc, float eps, int num_tokens, int num_heads_q,
                                  int64_t page_stride, int page_bits, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
