/* Copyright 2026 Tencent Inc.  All rights reserved.
 * Adapted from https://github.com/sgl-project/sglang/blob/f63def851/sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu
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

#include "csrc/kernels/nvidia/grouped_topk/topk_gating_softmax.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace llm_kernels {
namespace nvidia {

#define TOPK_SHFL_XOR(mask, var, lane_mask, width) __shfl_xor_sync((mask), (var), (lane_mask), (width))

static constexpr int kWarpSize = 32;

/// Aligned array for coalesced memory loads
template <typename T_Arr, int N_Arr, int Alignment_Arr = sizeof(T_Arr) * N_Arr>
class alignas(Alignment_Arr) TopkAlignedArray {
 public:
  T_Arr data[N_Arr];
};

/// Convert to float32
template <typename T>
__device__ __forceinline__ float topk_to_float(T x) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(x);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(x);
  } else {
    return static_cast<float>(x);
  }
}

// ============================================================================
// Core kernel: fused softmax + topk with warp-parallel butterfly reductions
// ============================================================================
template <typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA * kWarpSize) __global__ void TopkGatingSoftmaxKernel(
    const T* __restrict__ input, float* __restrict__ output, const int num_rows,
    int32_t* __restrict__ indices, const int k, const bool renormalize,
    const float routed_scaling_factor) {
  static_assert(VPT == (VPT & -VPT));
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS));
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG));
  static_assert(BYTES_PER_LDG <= 16);

  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  static_assert(VPT % ELTS_PER_LDG == 0);
  static_assert(kWarpSize % THREADS_PER_ROW == 0);
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW));
  static_assert(THREADS_PER_ROW <= kWarpSize);

  static constexpr int ELTS_PER_WARP = kWarpSize * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0);

  // Compute row assignment for this thread
  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;

  if (thread_row >= num_rows) return;

  // Setup read pointers with coalesced access pattern
  const T* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  // Coalesced vectorized load
  using AccessType = TopkAlignedArray<T, ELTS_PER_LDG>;
  T row_chunk_temp[VPT];
  AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk_temp);
  const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
  }

  // Convert to float32
  float row_chunk[VPT];
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = topk_to_float<T>(row_chunk_temp[ii]);
  }

  // ===== Softmax: find max via butterfly reduction =====
  float thread_max = row_chunk[0];
#pragma unroll
  for (int ii = 1; ii < VPT; ++ii) {
    thread_max = max(thread_max, row_chunk[ii]);
  }
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    thread_max = max(thread_max, TOPK_SHFL_XOR(0xffffffff, thread_max, mask, THREADS_PER_ROW));
  }

  // ===== Softmax: exp and sum =====
  float row_sum = 0;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = expf(row_chunk[ii] - thread_max);
    row_sum += row_chunk[ii];
  }
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    row_sum += TOPK_SHFL_XOR(0xffffffff, row_sum, mask, THREADS_PER_ROW);
  }

  // ===== Softmax: normalize =====
  const float reciprocal_row_sum = 1.f / row_sum;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] *= reciprocal_row_sum;
  }

  // ===== Top-k selection via iterative argmax =====
  const int start_col = first_elt_read_by_thread;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;
  float row_sum_for_renormalize = 0;

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    // Local argmax
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        float val = row_chunk[ldg * ELTS_PER_LDG + ii];
        if (val > max_val) {
          max_val = val;
          expert = col + ii;
        }
      }
    }

    // Butterfly argmax reduction across thread group
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max = TOPK_SHFL_XOR(0xffffffff, max_val, mask, THREADS_PER_ROW);
      int other_expert = TOPK_SHFL_XOR(0xffffffff, expert, mask, THREADS_PER_ROW);
      if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    // Write result (only lead thread per row)
    if (thread_group_idx == 0) {
      const int idx = k * thread_row + k_idx;
      output[idx] = max_val;
      indices[idx] = expert;
      row_sum_for_renormalize += max_val;
    }

    // Zero out selected expert for next iteration
    if (k_idx + 1 < k) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
      }
    }
  }

  // ===== Apply renormalize + routed_scaling_factor =====
  if (thread_group_idx == 0) {
    if (renormalize) {
      float scale = routed_scaling_factor / row_sum_for_renormalize;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        output[k * thread_row + k_idx] *= scale;
      }
    } else if (routed_scaling_factor != 1.0f) {
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        output[k * thread_row + k_idx] *= routed_scaling_factor;
      }
    }
  }
}

// ============================================================================
// Launch configuration helpers
// ============================================================================
namespace topk_detail {
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int VECs_PER_THREAD =
      (EXPERTS / (ELTS_PER_LDG * kWarpSize)) > 0 ? (EXPERTS / (ELTS_PER_LDG * kWarpSize)) : 1;
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = kWarpSize / THREADS_PER_ROW;
};
}  // namespace topk_detail

template <typename T, int EXPERTS, int WARPS_PER_TB>
void TopkGatingSoftmaxLaunch(const T* input, float* output, int32_t* indices, int num_rows, int k,
                             bool renormalize, float routed_scaling_factor, cudaStream_t stream) {
  static constexpr std::size_t MAX_BYTES_PER_LDG = 16;
  static constexpr int BYTES_PER_LDG =
      (static_cast<int>(MAX_BYTES_PER_LDG) < static_cast<int>(sizeof(T) * EXPERTS))
          ? static_cast<int>(MAX_BYTES_PER_LDG)
          : static_cast<int>(sizeof(T) * EXPERTS);
  using Constants = topk_detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;

  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;
  dim3 block_dim(kWarpSize, WARPS_PER_TB);

  TopkGatingSoftmaxKernel<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG>
      <<<num_blocks, block_dim, 0, stream>>>(input, output, num_rows, indices, k, renormalize, routed_scaling_factor);
}

#define LAUNCH_TOPK_GATING(T_TYPE, NUM_EXPERTS, WARPS)                                                 \
  TopkGatingSoftmaxLaunch<T_TYPE, NUM_EXPERTS, WARPS>(static_cast<const T_TYPE*>(gating_output),        \
      static_cast<float*>(topk_weights_ptr), static_cast<int32_t*>(topk_ids_ptr), num_rows, topk,      \
      renormalize, routed_scaling_factor, stream)

// ============================================================================
// Public API implementation
// ============================================================================
template <typename T>
void InvokeTopkGatingSoftmax(const void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr, int num_rows,
                             int num_experts, int topk, bool renormalize, float routed_scaling_factor,
                             cudaStream_t stream) {
  if (num_rows <= 0 || topk > num_experts) return;

  static constexpr int WARPS_PER_TB = 4;
  switch (num_experts) {
    case 1:   LAUNCH_TOPK_GATING(T, 1, WARPS_PER_TB);   break;
    case 2:   LAUNCH_TOPK_GATING(T, 2, WARPS_PER_TB);   break;
    case 4:   LAUNCH_TOPK_GATING(T, 4, WARPS_PER_TB);   break;
    case 8:   LAUNCH_TOPK_GATING(T, 8, WARPS_PER_TB);   break;
    case 16:  LAUNCH_TOPK_GATING(T, 16, WARPS_PER_TB);  break;
    case 32:  LAUNCH_TOPK_GATING(T, 32, WARPS_PER_TB);  break;
    case 64:  LAUNCH_TOPK_GATING(T, 64, WARPS_PER_TB);  break;
    case 128: LAUNCH_TOPK_GATING(T, 128, WARPS_PER_TB); break;
    case 256: LAUNCH_TOPK_GATING(T, 256, WARPS_PER_TB); break;
    default: break;  // Non-power-of-2 not supported by this kernel
  }
}

#undef LAUNCH_TOPK_GATING
#undef TOPK_SHFL_XOR

// Explicit instantiations
template void InvokeTopkGatingSoftmax<float>(const void*, void*, void*, int, int, int, bool, float, cudaStream_t);
template void InvokeTopkGatingSoftmax<half>(const void*, void*, void*, int, int, int, bool, float, cudaStream_t);
template void InvokeTopkGatingSoftmax<__nv_bfloat16>(const void*, void*, void*, int, int, int, bool, float,
                                                     cudaStream_t);

}  // namespace nvidia
}  // namespace llm_kernels
