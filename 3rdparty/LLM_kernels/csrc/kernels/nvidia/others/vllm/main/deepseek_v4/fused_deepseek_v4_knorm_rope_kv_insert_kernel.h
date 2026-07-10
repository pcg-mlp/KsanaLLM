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
 * @brief Fused weighted RMSNorm + RoPE + FP8 paged-cache insert for DSpark K injection.
 *
 * Applies a *weighted* RMSNorm (using kv_norm_weight) on the projected K input,
 * GPT-J style RoPE on the rope dims, FP8 quantization of the nope dims (with UE8M0
 * scales for kFp8DsMla, dense for kFp8E4M3), then writes the result into the paged
 * SWA KV cache slot pointed to by raw_out_loc[token].
 *
 * Dimensions:
 *   head_dim         = 512
 *   qk_nope_head_dim = 448
 *   qk_rope_head_dim = 64
 *
 * @param kv_in           [num_tokens, 512]  bf16, projected K input (output of wkv).
 * @param kv_norm_weight  [512]              bf16, kv_norm.weight applied after RMSNorm.
 * @param swa_kv_cache    Paged KV cache byte array (FP8_DS_MLA or FP8_E4M3 layout).
 * @param positions       [num_tokens]       int64, position ids for cos/sin lookup.
 * @param cos_sin_cache   [max_pos, 64]      bf16, layout [cos(32), sin(32)] per position.
 * @param raw_out_loc     [num_tokens]       int32, slot indices for SWA cache.
 *                        page = index >> page_bits, offset = index & page_mask.
 *                        -1 means skip store for that token.
 * @param eps             RMSNorm epsilon (e.g. 1e-6).
 * @param num_tokens      Total number of tokens (N).
 * @param page_stride     Stride in bytes between pages in swa_kv_cache.
 * @param page_bits       log2(page_size): 1, 6, or 8.
 * @param stream          CUDA stream.
 */
template <llm_kernels::utils::KVCacheType KV_DTYPE = llm_kernels::utils::KVCacheType::kFp8DsMla>
void InvokeFusedKNormRopeKVInsert(const __nv_bfloat16* kv_in, const __nv_bfloat16* kv_norm_weight, void* swa_kv_cache,
                                  const int64_t* positions, const __nv_bfloat16* cos_sin_cache,
                                  const int32_t* raw_out_loc, float eps, int num_tokens, int64_t page_stride,
                                  int page_bits, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
