/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================

This file contains code ported from vLLM.
Original source:
  - Repository: https://github.com/vllm-project/vllm
  - PR:         https://github.com/vllm-project/vllm/pull/40760
  - File:       csrc/moe/topk_softplus_sqrt_kernels.cu
  - Originally adapted by vLLM from
    https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

==============================================================================*/

#include "csrc/kernels/nvidia/others/vllm/main/topk_softplus_sqrt/topk_softplus_sqrt.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "csrc/utils/nvidia/assert.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace llm_kernels {
namespace nvidia {

namespace {

// Aligned array type
template <typename T, int N, int Alignment = sizeof(T) * N>
struct alignas(Alignment) AlignedArray {
  T data[N];
};

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(value);
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(value);
  }
}

#define FINAL_MASK 0xffffffff
template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

/*
  A Top-K gating softplus_sqrt written to exploit when the number of experts in
  the MoE layers are a small power of 2. This allows us to cleanly share the
  rows among the threads in a single warp and eliminate communication between
  warps (so no need to use shared mem).
*/
template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, int WARP_SIZE_PARAM, bool USE_HASH,
          typename IndType, typename InputType = float>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE_PARAM) __global__
    void topkGatingSoftplusSqrt(const InputType* input, const bool* finished, float* output, const int num_rows,
                                IndType* indices, int* source_rows, const int k, const int start_expert,
                                const int end_expert, const bool renormalize, double routed_scaling_factor,
                                const float* correction_bias, const IndType* input_ids, const IndType* tid2eid) {
  static_assert(
      std::is_same_v<InputType, float> || std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>,
      "InputType must be float, __nv_bfloat16, or __half");

  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  if constexpr (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>) {
    static_assert(ELTS_PER_LDG == 1 || ELTS_PER_LDG % 2 == 0, "ELTS_PER_LDG must be 1 or even for 16-bit conversion");
  }

  static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE_PARAM % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE_PARAM, "THREADS_PER_ROW can be at most warp size");

  static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;

  if (thread_row >= num_rows) {
    return;
  }
  const bool row_is_active = finished ? !finished[thread_row] : true;

  const InputType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const InputType* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  float row_chunk[VPT];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  if constexpr (std::is_same_v<InputType, float>) {
    using VecType = AlignedArray<float, ELTS_PER_LDG>;
    VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(&row_chunk);
    const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }
  } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
    if constexpr (ELTS_PER_LDG >= 2) {
      using VecType = AlignedArray<__nv_bfloat16, ELTS_PER_LDG>;
      float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
      const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
        for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
          row_chunk_f2[base_idx_f2 + jj] =
              __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(vec.data + jj * 2));
        }
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        const __nv_bfloat16* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
        row_chunk[ii] = __bfloat162float(*scalar_ptr);
      }
    }
  } else if constexpr (std::is_same_v<InputType, __half>) {
    if constexpr (ELTS_PER_LDG >= 2) {
      using VecType = AlignedArray<__half, ELTS_PER_LDG>;
      float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
      const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
        for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
          row_chunk_f2[base_idx_f2 + jj] = __half22float2(*reinterpret_cast<const __half2*>(vec.data + jj * 2));
        }
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        const __half* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
        row_chunk[ii] = __half2float(*scalar_ptr);
      }
    }
  }
  constexpr float threshold = 20.0f;
  constexpr float beta = 1.0f;

  // Hash MoE path: indices are predetermined from lookup table
  if constexpr (USE_HASH) {
    const IndType token_id = input_ids[thread_row];
    const IndType* expert_indices_for_token = tid2eid + token_id * k;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      float val = row_chunk[ii];
      float val_b = val * beta;
      val = (val_b > threshold) ? val : (__logf(1.0f + __expf(val_b))) / beta;
      row_chunk[ii] = sqrtf(val);
    }
    float selected_sum = 0.f;
#pragma unroll
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int expert = expert_indices_for_token[k_idx];
      const int idx = k * thread_row + k_idx;
      for (int ii = 0; ii < VPT; ++ii) {
        const int group_id = ii / ELTS_PER_LDG;
        const int local_id = ii % ELTS_PER_LDG;
        const int expert_idx = first_elt_read_by_thread + group_id * THREADS_PER_ROW * ELTS_PER_LDG + local_id;
        if (expert == expert_idx) {
          indices[idx] = expert;
          selected_sum += row_chunk[ii];
          break;
        }
      }
    }
    if (renormalize) {
      selected_sum = warpReduceSum(selected_sum);
    }
    float scale = static_cast<float>(routed_scaling_factor);
    if (renormalize) {
      const float denom = selected_sum > 0.f ? selected_sum : 1.f;
      scale /= denom;
    }

#pragma unroll
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int expert = expert_indices_for_token[k_idx];
      const int idx = k * thread_row + k_idx;
      for (int ii = 0; ii < VPT; ++ii) {
        const int group_id = ii / ELTS_PER_LDG;
        const int local_id = ii % ELTS_PER_LDG;
        const int expert_idx = first_elt_read_by_thread + group_id * THREADS_PER_ROW * ELTS_PER_LDG + local_id;
        if (expert == expert_idx) {
          output[idx] = row_chunk[ii] * scale;
          break;
        }
      }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    return;
  }

#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    float val = row_chunk[ii];
    float val_b = val * beta;
    val = (val_b > threshold) ? val : (__logf(1.0f + __expf(val_b))) / beta;
    val = sqrtf(val);
    if (correction_bias) {
      const int group_id = ii / ELTS_PER_LDG;
      const int local_id = ii % ELTS_PER_LDG;
      const int expert_idx = first_elt_read_by_thread + group_id * THREADS_PER_ROW * ELTS_PER_LDG + local_id;
      val = val + correction_bias[expert_idx];
    }
    row_chunk[ii] = val;
  }

  int start_col = first_elt_read_by_thread;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  float selected_sum = 0.f;
  for (int k_idx = 0; k_idx < k; ++k_idx) {
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

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max = __shfl_xor_sync(uint32_t(-1), max_val, mask, THREADS_PER_ROW);
      int other_expert = __shfl_xor_sync(uint32_t(-1), expert, mask, THREADS_PER_ROW);

      if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    if (thread_group_idx == 0) {
      const bool node_uses_expert = expert >= start_expert && expert < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;

      const int idx = k * thread_row + k_idx;
      if (correction_bias != nullptr) {
        max_val -= correction_bias[expert];
      }
      output[idx] = max_val;
      indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
      source_rows[idx] = k_idx * num_rows + thread_row;
      if (renormalize) {
        selected_sum += max_val;
      }
    }

    if (k_idx + 1 < k) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
      }
    }
  }

  if (thread_group_idx == 0) {
    float scale = static_cast<float>(routed_scaling_factor);
    if (renormalize) {
      const float denom = selected_sum > 0.f ? selected_sum : 1.f;
      scale /= denom;
    }
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int idx = k * thread_row + k_idx;
      output[idx] = output[idx] * scale;
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

namespace detail {
template <int EXPERTS, int BYTES_PER_LDG, int WARP_SIZE_PARAM, typename InputType>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0, "");
  static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static const int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};
}  // namespace detail

#define DISPATCH_HASH(use_hash, USE_HASH, ...)                                  \
  if (use_hash) {                                                               \
    const bool USE_HASH = true;                                                 \
    static_assert(USE_HASH == true, "USE_HASH must be compile-time constant");  \
    __VA_ARGS__                                                                 \
  } else {                                                                      \
    const bool USE_HASH = false;                                                \
    static_assert(USE_HASH == false, "USE_HASH must be compile-time constant"); \
    __VA_ARGS__                                                                 \
  }

template <int EXPERTS, int WARPS_PER_TB, int WARP_SIZE_PARAM, int MAX_BYTES_PER_LDG, typename IndType,
          typename InputType>
void topkGatingSoftplusSqrtLauncherHelper(const InputType* input, const bool* finished, float* output, IndType* indices,
                                          int* source_row, const int num_rows, const int k, const int start_expert,
                                          const int end_expert, const bool renormalize, double routed_scaling_factor,
                                          const float* correction_bias, const bool use_hash, const IndType* input_ids,
                                          const IndType* tid2eid, cudaStream_t stream) {
  static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(InputType) * EXPERTS);
  using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;
  dim3 block_dim(WARP_SIZE_PARAM, WARPS_PER_TB);
  DISPATCH_HASH(use_hash, USE_HASH, {
    auto* kernel = &topkGatingSoftplusSqrt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG, WARP_SIZE_PARAM, USE_HASH,
                                           IndType, InputType>;
    cudaLaunchConfig_t config = {};
    config.gridDim = num_blocks;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, input, finished, output, num_rows, indices, source_row, k, start_expert,
                       end_expert, renormalize, routed_scaling_factor, correction_bias, input_ids, tid2eid);
  })
}

#define LAUNCH_SOFTPLUS_SQRT(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)                                                \
  topkGatingSoftplusSqrtLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, 32, MAX_BYTES>(                                 \
      gating_output, nullptr, topk_weights, topk_indices, token_expert_indices, num_tokens, topk, 0, num_experts, \
      renormalize, routed_scaling_factor, correction_bias, use_hash, input_ids, tid2eid, stream);

template <typename IndType, typename InputType>
void topkGatingSoftplusSqrtKernelLauncher(const InputType* gating_output, float* topk_weights, IndType* topk_indices,
                                          int* token_expert_indices, const int num_tokens, const int num_experts,
                                          const int topk, const bool renormalize, double routed_scaling_factor,
                                          const float* correction_bias, const bool use_hash, const IndType* input_ids,
                                          const IndType* tid2eid, cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;
  static constexpr int BYTES_PER_LDG_POWER_OF_2 = 16;
  // For bfloat16/half dtype, we need 4 bytes loading to make sure num_experts
  // elements can be loaded by a warp
  static constexpr int BYTES_PER_LDG_MULTIPLE_64 =
      (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>) ? 4 : 8;
  switch (num_experts) {
    case 1:
      LAUNCH_SOFTPLUS_SQRT(1, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 2:
      LAUNCH_SOFTPLUS_SQRT(2, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 4:
      LAUNCH_SOFTPLUS_SQRT(4, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 8:
      LAUNCH_SOFTPLUS_SQRT(8, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 16:
      LAUNCH_SOFTPLUS_SQRT(16, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 32:
      LAUNCH_SOFTPLUS_SQRT(32, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 64:
      LAUNCH_SOFTPLUS_SQRT(64, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 128:
      LAUNCH_SOFTPLUS_SQRT(128, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 256:
      LAUNCH_SOFTPLUS_SQRT(256, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 512:
      LAUNCH_SOFTPLUS_SQRT(512, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 192:
      LAUNCH_SOFTPLUS_SQRT(192, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 320:
      LAUNCH_SOFTPLUS_SQRT(320, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 384:
      LAUNCH_SOFTPLUS_SQRT(384, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 448:
      LAUNCH_SOFTPLUS_SQRT(448, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 576:
      LAUNCH_SOFTPLUS_SQRT(576, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    default:
      KLLM_KERNEL_THROW("Unsupported num_experts: %d", num_experts);
  }
}

template <typename InputType>
void dispatchIndexType(const InputType* gating_output, void* topk_weights, void* topk_indices,
                       void* token_expert_indices, int num_tokens, int num_experts, int topk, bool renormalize,
                       double routed_scaling_factor, const float* correction_bias, const void* input_ids,
                       const void* tid2eid, tensorrt_llm::dev::ScalarType index_dtype, cudaStream_t stream) {
  const bool use_hash = (tid2eid != nullptr);
  switch (index_dtype) {
    case tensorrt_llm::dev::ScalarType::Int:
      topkGatingSoftplusSqrtKernelLauncher<int32_t, InputType>(
          gating_output, static_cast<float*>(topk_weights), static_cast<int32_t*>(topk_indices),
          static_cast<int*>(token_expert_indices), num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
          correction_bias, use_hash, static_cast<const int32_t*>(input_ids), static_cast<const int32_t*>(tid2eid),
          stream);
      break;
    case tensorrt_llm::dev::ScalarType::Long:
      topkGatingSoftplusSqrtKernelLauncher<int64_t, InputType>(
          gating_output, static_cast<float*>(topk_weights), static_cast<int64_t*>(topk_indices),
          static_cast<int*>(token_expert_indices), num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
          correction_bias, use_hash, static_cast<const int64_t*>(input_ids), static_cast<const int64_t*>(tid2eid),
          stream);
      break;
    default:
      KLLM_KERNEL_THROW("InvokeTopkSoftplusSqrt: unsupported index_dtype %d (expected Int or Long)",
                        static_cast<int>(index_dtype));
  }
}

}  // anonymous namespace

void InvokeTopkSoftplusSqrt(const void* gating_output, void* topk_weights, void* topk_indices,
                            void* token_expert_indices, int num_tokens, int num_experts, int topk, bool renormalize,
                            double routed_scaling_factor, const float* correction_bias, const void* input_ids,
                            const void* tid2eid, tensorrt_llm::dev::ScalarType input_dtype,
                            tensorrt_llm::dev::ScalarType index_dtype, cudaStream_t stream) {
  if (num_tokens == 0) return;

  switch (input_dtype) {
    case tensorrt_llm::dev::ScalarType::Float:
      dispatchIndexType<float>(static_cast<const float*>(gating_output), topk_weights, topk_indices,
                               token_expert_indices, num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
                               correction_bias, input_ids, tid2eid, index_dtype, stream);
      break;
    case tensorrt_llm::dev::ScalarType::Float16:
      dispatchIndexType<__half>(static_cast<const __half*>(gating_output), topk_weights, topk_indices,
                                token_expert_indices, num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
                                correction_bias, input_ids, tid2eid, index_dtype, stream);
      break;
    case tensorrt_llm::dev::ScalarType::BFloat16:
      dispatchIndexType<__nv_bfloat16>(static_cast<const __nv_bfloat16*>(gating_output), topk_weights, topk_indices,
                                       token_expert_indices, num_tokens, num_experts, topk, renormalize,
                                       routed_scaling_factor, correction_bias, input_ids, tid2eid, index_dtype, stream);
      break;
    default:
      KLLM_KERNEL_THROW("InvokeTopkSoftplusSqrt: unsupported input_dtype %d (expected Float / Float16 / BFloat16)",
                        static_cast<int>(input_dtype));
  }
}

}  // namespace nvidia
}  // namespace llm_kernels
