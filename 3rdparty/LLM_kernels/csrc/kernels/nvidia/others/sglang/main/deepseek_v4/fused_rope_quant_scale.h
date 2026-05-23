/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {


// Fused RoPE + FP8 quantization + weight scaling.
//
// 1. RoPE rotation on 64 dims of Q at the given rope_offset within the head.
// 2. Per-head FP8 (E4M3) abs-max quantization of the full head.
// 3. output_scale = q_scale * weight * (n_heads_inv_sqrt * softmax_scale)
//
// q:            [batch * heads, 128] bf16, contiguous.
// weights:      [batch * heads] bf16, contiguous.
// freqs_cis:    [max_pos, 64] bf16, split [cos|sin].
// positions:    [batch] int64.
// output_q_fp8: [batch * heads, 128] fp8_e4m3, contiguous.
// output_scale: [batch * heads] f32.
// rope_offset:  element offset within head where the 64 rope dims start.
//               - 0  for [rope_dim, nope_dim] layout (DSA indexer)
//               - 64 for [nope_dim, rope_dim] layout (C4 indexer)
//
void fused_rope_quant_scale(const void* q, const void* weights, const void* freqs_cis, const void* positions,
                            void* output_q_fp8, float* output_scale, float n_heads_inv_sqrt, float softmax_scale,
                            uint32_t num_heads, uint32_t batch_size, uint32_t rope_offset, cudaStream_t stream,
                            bool is_neox = false);

}  // namespace nvidia
}  // namespace llm_kernels
