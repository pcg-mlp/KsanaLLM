/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {


// DeepSeek-V4 in-place RoPE for Q and K jointly.
//
// Fixed dtype: __nv_bfloat16; fixed kRopeDim=64 (each warp lane handles one
// complex pair).  The last kRopeDim elements of each head are rotated.
//
// Layout of q: [batch_size, num_q_heads, kRopeDim] with strides
//   q_stride_batch / q_stride_head in **bytes**.
// Layout of k: [batch_size, num_k_heads, kRopeDim] with strides
//   k_stride_batch / k_stride_head in **bytes**.
// freqs_cis : [-1, kRopeDim] bf16 (split layout: first half cos, second half sin).
// positions : [batch_size] as int64_t.
//
// `inverse` reverses the rotation: (a+bi)(c-di) instead of (a+bi)(c+di).
//
// q/k can be null when the corresponding num_heads == 0.
void fused_qk_rope(void* q, void* k, const void* freqs_cis, const void* positions, int64_t q_stride_batch,
                   int64_t k_stride_batch, int64_t q_stride_head, int64_t k_stride_head, uint32_t num_q_heads,
                   uint32_t num_k_heads, uint32_t batch_size, cudaStream_t stream, bool inverse = false,
                   bool is_neox = false);

}  // namespace nvidia
}  // namespace llm_kernels
