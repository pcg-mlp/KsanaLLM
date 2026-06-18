/* Copyright 2026 Tencent Inc.  All rights reserved.
 * Adapted from https://github.com/sgl-project/sglang/blob/9858113c/sgl-kernel/csrc/moe/moe_sum_reduce.cu
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ==============================================================================*/

#pragma once

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

/**
 * @brief Vectorized MoE sum reduction (sglang-derived)
 *
 * Sums expert outputs [num_tokens, topk, hidden_size] -> [num_tokens, hidden_size]
 * with optional scaling factor, using uint4 vectorized loads for BF16/FP16.
 *
 * ~2-3x faster than the scalar moe_sum_kernel for typical hidden_size (2048+).
 */
template <typename T>
void InvokeMoeSumVec(const void* input,    // [num_tokens, topk, hidden_size]
                     void* output,          // [num_tokens, hidden_size]
                     int num_tokens, int topk, int hidden_size,
                     float routed_scaling_factor, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
