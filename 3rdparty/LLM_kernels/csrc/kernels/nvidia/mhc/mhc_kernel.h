/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {

// Post big-fuse:
//   out[t, hc, d] = x[t, d] * post_layer_mix[t, hc]
//                   + sum_{j=0..3} comb_res_mix[t, j, hc] * residual[t, j, d]
// hc fixed at 4. d must be > 0.
// When d % 8 == 0 a vec8 path is used; otherwise a scalar fallback runs.
void mhc_post(void* out,                    // [t, hc, d]
              const void* x,                // [t, d]
              const void* residual,         // [t, hc, d]
              const float* post_layer_mix,  // [t, hc]
              const float* comb_res_mix,    // [t, hc, hc]
              size_t total_tokens, size_t d, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
