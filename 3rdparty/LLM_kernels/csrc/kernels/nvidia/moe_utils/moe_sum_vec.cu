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

#include "csrc/kernels/nvidia/moe_utils/moe_sum_vec.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <type_traits>

namespace llm_kernels {
namespace nvidia {

constexpr int kWarpSize = 32;

// ============================================================================
// Vectorized kernel: 16-byte loads (8 x BF16/FP16), float32 accumulation
// Each warp handles one token, threads cooperate on hidden_dim chunks.
// ============================================================================
template <typename T, int WARPS_PER_BLOCK>
__global__ void MoeSumVecKernel(const T* __restrict__ input, T* __restrict__ output,
                                const int64_t num_tokens, const int64_t hidden_dim,
                                const int64_t topk, const float scale) {
  constexpr int VEC_ELTS = 16 / sizeof(T);  // 16 bytes total per vector load

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t token_idx = static_cast<int64_t>(blockIdx.y) * WARPS_PER_BLOCK + warp_id;
  if (token_idx >= num_tokens) return;

  const int64_t n_chunks = hidden_dim / VEC_ELTS;
  const int64_t stride_token = topk * hidden_dim;
  const int64_t stride_topk = hidden_dim;

  for (int64_t chunk = static_cast<int64_t>(blockIdx.x) * kWarpSize + lane; chunk < n_chunks;
       chunk += static_cast<int64_t>(gridDim.x) * kWarpSize) {
    const int64_t d = chunk * VEC_ELTS;
    const int64_t base = token_idx * stride_token + d;

    float acc[VEC_ELTS];
#pragma unroll
    for (int i = 0; i < VEC_ELTS; ++i) acc[i] = 0.f;

    // Accumulate across topk experts with vectorized loads
    for (int k = 0; k < topk; ++k) {
      const int64_t offset = base + static_cast<int64_t>(k) * stride_topk;
      uint4 packed = *reinterpret_cast<const uint4*>(input + offset);
      const T* elts = reinterpret_cast<const T*>(&packed);
#pragma unroll
      for (int i = 0; i < VEC_ELTS; ++i) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
          acc[i] += __bfloat162float(elts[i]);
        } else {
          acc[i] += __half2float(elts[i]);
        }
      }
    }

    // Apply scale and write back
#pragma unroll
    for (int i = 0; i < VEC_ELTS; ++i) acc[i] *= scale;

    // Pack and store
    uint4 out_packed;
    T* out_elts = reinterpret_cast<T*>(&out_packed);
#pragma unroll
    for (int i = 0; i < VEC_ELTS; ++i) {
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        out_elts[i] = __float2bfloat16_rn(acc[i]);
      } else {
        out_elts[i] = __float2half_rn(acc[i]);
      }
    }
    *reinterpret_cast<uint4*>(output + token_idx * hidden_dim + d) = out_packed;
  }
}

// ============================================================================
// Scalar fallback for float32 or non-aligned hidden_dim
// ============================================================================
template <typename T, int WARPS_PER_BLOCK>
__global__ void MoeSumScalarKernel(const T* __restrict__ input, T* __restrict__ output,
                                   const int64_t num_tokens, const int64_t hidden_dim,
                                   const int64_t topk, const float scale) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t token_idx = static_cast<int64_t>(blockIdx.y) * WARPS_PER_BLOCK + warp_id;
  if (token_idx >= num_tokens) return;

  const int64_t stride_token = topk * hidden_dim;
  const int64_t stride_topk = hidden_dim;

  for (int64_t d = static_cast<int64_t>(blockIdx.x) * kWarpSize + lane; d < hidden_dim;
       d += static_cast<int64_t>(gridDim.x) * kWarpSize) {
    float acc = 0.f;
    const int64_t base = token_idx * stride_token + d;
    for (int k = 0; k < topk; ++k) {
      acc += static_cast<float>(input[base + static_cast<int64_t>(k) * stride_topk]);
    }
    acc *= scale;
    output[token_idx * hidden_dim + d] = static_cast<T>(acc);
  }
}

// ============================================================================
// Public API
// ============================================================================
template <typename T>
void InvokeMoeSumVec(const void* input, void* output, int num_tokens, int topk, int hidden_size,
                     float routed_scaling_factor, cudaStream_t stream) {
  if (num_tokens <= 0) return;

  constexpr int WARPS_PER_BLOCK = 4;
  constexpr int THREADS = WARPS_PER_BLOCK * kWarpSize;

  // Use vectorized path for BF16/FP16 with aligned hidden_size (16-byte alignment)
  const bool use_vec = (sizeof(T) == 2) && (hidden_size % (16 / sizeof(T)) == 0);

  if (use_vec) {
    const int64_t vec_elts = 16 / sizeof(T);
    const int64_t n_chunks = hidden_size / vec_elts;
    int grid_x = static_cast<int>(std::min(static_cast<int64_t>((n_chunks + kWarpSize - 1) / kWarpSize), static_cast<int64_t>(65535)));
    int grid_y = static_cast<int>(
        std::min(static_cast<int64_t>((num_tokens + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK),
                 static_cast<int64_t>(65535)));
    dim3 grid(grid_x, grid_y);
    dim3 block(THREADS);

    MoeSumVecKernel<T, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), num_tokens, hidden_size, topk,
        routed_scaling_factor);
  } else {
    int grid_x = static_cast<int>(std::min(static_cast<int64_t>((hidden_size + kWarpSize - 1) / kWarpSize), static_cast<int64_t>(65535)));
    int grid_y = static_cast<int>(
        std::min(static_cast<int64_t>((num_tokens + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK),
                 static_cast<int64_t>(65535)));
    dim3 grid(grid_x, grid_y);
    dim3 block(THREADS);

    MoeSumScalarKernel<T, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), num_tokens, hidden_size, topk,
        routed_scaling_factor);
  }
}

// Explicit instantiations
template void InvokeMoeSumVec<float>(const void*, void*, int, int, int, float, cudaStream_t);
template void InvokeMoeSumVec<half>(const void*, void*, int, int, int, float, cudaStream_t);
template void InvokeMoeSumVec<__nv_bfloat16>(const void*, void*, int, int, int, float, cudaStream_t);

}  // namespace nvidia
}  // namespace llm_kernels
