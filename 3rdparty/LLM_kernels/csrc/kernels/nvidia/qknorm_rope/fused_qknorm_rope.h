/* Copyright 2026 Tencent Inc.  All rights reserved.
 ==============================================================================*/

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {

void InvokeFusedQKNormRoPE(const void* input, void* output, const void* norm_weight,
                           const void* positions, const void* cos_sin_cache,
                           int num_tokens, int num_heads, int head_dim, float eps,
                           cudaStream_t stream, int stride_size = 0,
                           const int64_t* mask = nullptr);

}  // namespace nvidia
}  // namespace llm_kernels
