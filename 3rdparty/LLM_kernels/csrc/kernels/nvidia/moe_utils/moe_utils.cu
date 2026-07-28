/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 * Copyright 2025 vLLM Team
 * Copyright (c) 2023 DeepSeek
 * Copyright 2023-2024 SGLang Team
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
 *
 * Adapted from
 * [vLLM Project]
 * https://github.com/vllm-project/vllm/blob/09e56f92620908d3cf1c3020336460f0db8beead/csrc/moe/moe_align_sum_kernels.cu
 * [vLLM Project]
 * https://github.com/vllm-project/vllm/blob/v0.7.1/vllm/model_executor/layers/quantization/utils/quant_utils.py#L63
 * [DeepSeek-V3 Project] https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py#L84
 * [SGLang Project] https://github.com/sgl-project/sglang
 */
#include "moe_utils.h"

#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <flashinfer/activation.cuh>

#include "csrc/kernels/nvidia/common/vec_dtypes.cuh"
#include "csrc/utils/nvidia/cub_compat.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

#define CEILDIV(x, y) (((x) + (y)-1) / (y))
#define WARP_SIZE 32

using FP8_TYPE = uint8_t;
inline constexpr float FP8_E4M3_MAX = 448.0f;

namespace {
__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row, int32_t col) {
  // don't worry about overflow because num_experts is relatively small
  return row * total_col + col;
}
}  // namespace

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                     : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

// Vectorization containers
template <typename scalar_t>
struct __align__(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

template <typename quant_type_t>
struct __align__(4) q8x4_t {
  static_assert(std::is_same_v<quant_type_t, int8_t> || std::is_same_v<quant_type_t, uint8_t>);
  quant_type_t x;
  quant_type_t y;
  quant_type_t z;
  quant_type_t w;
};

template <bool is_scale_inverted>
__device__ __forceinline__ FP8_TYPE scaled_fp8_conversion(float const val, float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
  // NOTE: FP8_TYPE is uint8_t; a plain static_cast<uint8_t>(float) does INTEGER
  // truncation (e.g. 127.5 → 127), which collides with E4M3 NaN encoding (0x7F).
  // Use CUDA's proper float-to-FP8 conversion intrinsic to get the actual FP8 bits.
  return static_cast<FP8_TYPE>(__nv_cvt_float_to_fp8(r, __NV_SATFINITE, __NV_E4M3));
}

template <typename scalar_t, bool is_scale_inverted>
__device__ void scaled_fp8_conversion_vec(void* __restrict__ out, scalar_t const* __restrict__ input, float const scale,
                                          int64_t const num_elems, int const tid, int const step);

template <typename scalar_t, typename token_cnts_t>
__global__ void moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                            int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
                                            int32_t block_size, size_t numel) {
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;  // 1d tensor with shape (num_experts + 1)
  token_cnts_t* tokens_cnts =
      (token_cnts_t*)(shared_mem + num_experts + 1);  // 2d tensor with shape (blockDim.x + 1, num_experts)

  for (int32_t i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  /**
   * In the first step we compute token_cnts[thread_index + 1][expert_index],
   * which counts how many tokens in the token shard of thread_index are
   * assigned to expert expert_index.
   */
  for (size_t i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    ++tokens_cnts[index(num_experts, threadIdx.x + 1, topk_ids[i])];
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  if (threadIdx.x < num_experts) {
    tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
    for (unsigned int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[index(num_experts, i, threadIdx.x)] += tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
    }
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int32_t i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] + CEILDIV(tokens_cnts[index(num_experts, blockDim.x, i - 1)], block_size) * block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  if (threadIdx.x < num_experts) {
    for (int32_t i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  /**
   * Each thread processes a token shard, calculating the index of each token
   * after sorting by expert number. Given the example topk_ids =
   * [0,1,2,1,2,3,0,3,4] and block_size = 4, then the output would be [0, 6, *,
   * *, 1, 3, *, *, 2, 4, *, *, 5, 7, *, *, 8, *, *, *], where * represents a
   * padding value(preset in python).
   */
  for (size_t i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    /** The cumsum[expert_id] stores the starting index of the tokens that the
     * expert with expert_id needs to process, and
     * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
     * processed by the expert with expert_id within the current thread's token
     * shard.
     */
    int32_t rank_post_pad = tokens_cnts[index(num_experts, threadIdx.x, expert_id)] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
  }
}

template <typename scalar_t, typename token_cnts_t>
__global__ void moe_align_block_size_kernel_expert_parallel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                                            int32_t* expert_ids, int32_t* total_tokens_post_pad,
                                                            int32_t topk, int32_t num_experts, int32_t block_size,
                                                            size_t numel) {
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;  // 1d tensor with shape (num_experts + 1)
  token_cnts_t* tokens_cnts =
      (token_cnts_t*)(shared_mem + num_experts + 1);  // 2d tensor with shape (blockDim.x + 1, num_experts)

  // 把共享内存从 threadIdx.x * num_experts 开始， expert_num 个数字置 0
  for (int32_t i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  /**
   * In the first step we compute token_cnts[thread_index + 1][expert_index],
   * which counts how many tokens in the token shard of thread_index are
   * assigned to expert expert_index.
   */
  // 统计所有的 topk_ids，将对应的位置计数 +n
  for (size_t i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    if (expert_id >= 0) {
      ++tokens_cnts[index(num_experts, threadIdx.x + 1, expert_id)];
    }
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  // 我们用不同的 thread，来累加不同的专家
  if (threadIdx.x < num_experts) {
    tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
    for (unsigned int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[index(num_experts, i, threadIdx.x)] += tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
    }
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int32_t i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] + CEILDIV(tokens_cnts[index(num_experts, blockDim.x, i - 1)], block_size) * block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  if (threadIdx.x < num_experts) {
    for (int32_t i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  /**
   * Each thread processes a token shard, calculating the index of each token
   * after sorting by expert number. Given the example topk_ids =
   * [0,1,2,1,2,3,0,3,4] and block_size = 4, then the output would be [0, 6, *,
   * *, 1, 3, *, *, 2, 4, *, *, 5, 7, *, *, 8, *, *, *], where * represents a
   * padding value(preset in python).
   */
  for (size_t i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    if (expert_id < 0) {
      continue;
    }
    /** The cumsum[expert_id] stores the starting index of the tokens that the
     * expert with expert_id needs to process, and
     * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
     * processed by the expert with expert_id within the current thread's token
     * shard.
     */
    int32_t rank_post_pad = tokens_cnts[index(num_experts, threadIdx.x, expert_id)] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
  }
}

// taken from
// https://github.com/sgl-project/sglang/blame/f024795e57c3589a63df2457d3d64771989d4ed7/sgl-kernel/csrc/moe/moe_align_kernel.cu#L30
template <typename scalar_t, bool UseExpertParallel>
__global__ void sgl_moe_token_sort_kernel(const scalar_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
                                          int32_t* __restrict__ cumsum, size_t numel) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if constexpr (UseExpertParallel) {
      if (expert_id < 0) {
        continue;
      }
    }
    int32_t rank_post_pad = atomicAdd(&cumsum[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

__device__ __forceinline__ int warp_exclusive_scan(int v, const unsigned mask = 0xffffffffu) {
  int original = v;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int n = __shfl_up_sync(mask, v, offset);
    if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) {
      v += n;
    }
  }
  return v - original;
}

// taken from
// https://github.com/sgl-project/sglang/blame/f024795e57c3589a63df2457d3d64771989d4ed7/sgl-kernel/csrc/moe/moe_align_kernel.cu#L58
template <typename scalar_t, bool UseExpertParallel>
__global__ void sgl_moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, const int32_t max_num_tokens_padded, const int32_t num_experts,
    const int32_t block_size, const size_t numel, int32_t* __restrict__ cumsum, const int32_t scan_size) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  // Threads in blocks with indices `> 0` just perform `sorted_ids.fill_(numel)`
  if (blockIdx.x > 0) {
    const int32_t fill_val = static_cast<int32_t>(numel);
    vec4_t<int32_t> fill_vec;
    fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = fill_val;

    const int32_t stride = (gridDim.x - 1) * blockDim.x;

    // Vectorization
    const int32_t max_num_tokens_padded_vecs = max_num_tokens_padded / SGL_MOE_ALIGN_BLOCK_VEC_SIZE;
    vec4_t<int32_t>* out_ptr = reinterpret_cast<vec4_t<int32_t>*>(sorted_token_ids);
    for (int32_t i = (blockIdx.x - 1) * blockDim.x + threadIdx.x; i < max_num_tokens_padded_vecs; i += stride) {
      out_ptr[i] = fill_vec;
    }
    // Handle the remaining part
    for (int32_t i = max_num_tokens_padded_vecs * SGL_MOE_ALIGN_BLOCK_VEC_SIZE + threadIdx.x; i < max_num_tokens_padded;
         i += stride) {
      sorted_token_ids[i] = fill_val;
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    return;
  }

  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;                  // [num_experts]
  int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
  int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
  __shared__ int32_t s_total_tokens_post_pad;

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  if (tid < num_experts) {
    shared_counts[tid] = 0;
  }
  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if constexpr (UseExpertParallel) {
      if (expert_id < 0) {
        continue;
      }
    }
    atomicAdd(&shared_counts[expert_id], 1);
  }
  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count = (count + block_size - 1) / block_size * block_size;
    scan_buf[tid] = padded_count;
  }

  // Intra warp prefix sum
  int32_t* warp_sums = scan_buf + scan_size;  // [<= 32]
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid & (WARP_SIZE - 1);
  const int num_warps_for_scan = (scan_size + WARP_SIZE - 1) / WARP_SIZE;
  const int warp_sum = warp_exclusive_scan(padded_count) + padded_count;
  if (lane_id == WARP_SIZE - 1) {
    warp_sums[warp_id] = warp_sum;
  }
  __syncthreads();

  // warp0 accumulate all the block's prefix sum
  if (tid < WARP_SIZE) {
    int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
    int incl = warp_exclusive_scan(val) + val;
    warp_sums[tid] = incl;
  }
  __syncthreads();

  // Every thread obtains the whole block's sum
  if (tid == 0) {
    prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

  // Fill 0 to scan_buf extended area (tid >= num_expert)
  if (tid >= num_experts && tid < scan_size) {
    scan_buf[tid] = 0;
  }
  __syncthreads();

  // Perform 2 level exclusive-prefix-sum to scan_buf
  int v = (tid < scan_size) ? scan_buf[tid] : 0;
  int pre = warp_exclusive_scan(v);
  if (lane_id == WARP_SIZE - 1) {
    warp_sums[warp_id] = pre + v;
  }
  __syncthreads();

  if (warp_id == 0) {
    int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
    warp_sums[lane_id] = warp_exclusive_scan(val);
  }
  __syncthreads();

  int offset = warp_sums[warp_id];
  if (tid < scan_size) {
    scan_buf[tid] = pre + offset;
  }
  __syncthreads();

  // Write prefix[0..num_experts - 1] and cumsum
  if (tid < num_experts) {
    prefix[tid] = scan_buf[tid];
  }

  if (tid <= num_experts) {
    cumsum[tid] = prefix[tid];
  }

  // fill expert_ids
  const int32_t num_blocks = s_total_tokens_post_pad / block_size;
  for (int32_t i = tid; i < num_blocks; i += stride) {
    int32_t block_start = i * block_size;
    int left = 0, right = num_experts;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (prefix[mid] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    expert_ids[i] = left - 1;
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Merged align kernel: computes alignment for two block_m values in a single launch.
// shared_counts (per-expert token counts) is block_m-independent and computed once.
// cumsum and expert_ids are computed separately for each block_m.
// sorted_token_ids fill (padding with numel) is identical for both, done once for output1.
template <typename scalar_t, bool UseExpertParallel>
__global__ void sgl_moe_align_block_size_merged_kernel(
    const scalar_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids1, int32_t* __restrict__ expert_ids1,
    int32_t* __restrict__ total_tokens_post_pad1, int32_t* __restrict__ sorted_token_ids2,
    int32_t* __restrict__ expert_ids2, int32_t* __restrict__ total_tokens_post_pad2,
    const int32_t max_num_tokens_padded1, const int32_t max_num_tokens_padded2, const int32_t num_experts,
    const int32_t block_size1, const int32_t block_size2, const size_t numel, int32_t* __restrict__ cumsum1,
    int32_t* __restrict__ cumsum2, const int32_t scan_size) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  // Threads in blocks with indices `> 0` just perform fill for both outputs
  if (blockIdx.x > 0) {
    const int32_t fill_val = static_cast<int32_t>(numel);
    vec4_t<int32_t> fill_vec;
    fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = fill_val;

    const int32_t stride = (gridDim.x - 1) * blockDim.x;
    const int32_t max_padded = max(max_num_tokens_padded1, max_num_tokens_padded2);

    // Fill output1
    const int32_t vecs1 = max_num_tokens_padded1 / SGL_MOE_ALIGN_BLOCK_VEC_SIZE;
    vec4_t<int32_t>* out1_ptr = reinterpret_cast<vec4_t<int32_t>*>(sorted_token_ids1);
    for (int32_t i = (blockIdx.x - 1) * blockDim.x + threadIdx.x; i < vecs1; i += stride) {
      out1_ptr[i] = fill_vec;
    }
    for (int32_t i = vecs1 * SGL_MOE_ALIGN_BLOCK_VEC_SIZE + threadIdx.x; i < max_num_tokens_padded1; i += stride) {
      sorted_token_ids1[i] = fill_val;
    }

    // Fill output2 (if different buffer)
    if (sorted_token_ids2 != sorted_token_ids1) {
      const int32_t vecs2 = max_num_tokens_padded2 / SGL_MOE_ALIGN_BLOCK_VEC_SIZE;
      vec4_t<int32_t>* out2_ptr = reinterpret_cast<vec4_t<int32_t>*>(sorted_token_ids2);
      for (int32_t i = (blockIdx.x - 1) * blockDim.x + threadIdx.x; i < vecs2; i += stride) {
        out2_ptr[i] = fill_vec;
      }
      for (int32_t i = vecs2 * SGL_MOE_ALIGN_BLOCK_VEC_SIZE + threadIdx.x; i < max_num_tokens_padded2; i += stride) {
        sorted_token_ids2[i] = fill_val;
      }
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    return;
  }

  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;                   // [num_experts] — shared between both block_m
  int32_t* prefix1 = shared_counts + num_experts;  // [num_experts + 1]
  int32_t* scan_buf1 = prefix1 + num_experts + 1;  // [scan_size]
  int32_t* prefix2 = scan_buf1 + scan_size;        // [num_experts + 1]
  int32_t* scan_buf2 = prefix2 + num_experts + 1;  // [scan_size]
  int32_t* warp_sums = scan_buf2 + scan_size;      // [<= 32] — reused for both scans
  __shared__ int32_t s_total_tokens_post_pad1;
  __shared__ int32_t s_total_tokens_post_pad2;

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  // Phase 1: Count tokens per expert (block_m-independent, computed once)
  if (tid < num_experts) {
    shared_counts[tid] = 0;
  }
  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if constexpr (UseExpertParallel) {
      if (expert_id < 0) {
        continue;
      }
    }
    atomicAdd(&shared_counts[expert_id], 1);
  }
  __syncthreads();

  // Phase 2: Compute cumsum for block_size1
  int32_t padded_count1 = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count1 = (count + block_size1 - 1) / block_size1 * block_size1;
    scan_buf1[tid] = padded_count1;
  }

  // Intra warp prefix sum for block_size1
  {
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int num_warps_for_scan = (scan_size + WARP_SIZE - 1) / WARP_SIZE;
    const int warp_sum = warp_exclusive_scan(padded_count1) + padded_count1;
    if (lane_id == WARP_SIZE - 1) {
      warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
      int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
      int incl = warp_exclusive_scan(val) + val;
      warp_sums[tid] = incl;
    }
    __syncthreads();

    if (tid == 0) {
      prefix1[num_experts] = warp_sums[num_warps_for_scan - 1];
      s_total_tokens_post_pad1 = prefix1[num_experts];
      *total_tokens_post_pad1 = s_total_tokens_post_pad1;
    }
    __syncthreads();

    if (tid >= num_experts && tid < scan_size) {
      scan_buf1[tid] = 0;
    }
    __syncthreads();

    int v = (tid < scan_size) ? scan_buf1[tid] : 0;
    int pre = warp_exclusive_scan(v);
    if (lane_id == WARP_SIZE - 1) {
      warp_sums[warp_id] = pre + v;
    }
    __syncthreads();

    if (warp_id == 0) {
      int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
      warp_sums[lane_id] = warp_exclusive_scan(val);
    }
    __syncthreads();

    int offset = warp_sums[warp_id];
    if (tid < scan_size) {
      scan_buf1[tid] = pre + offset;
    }
    __syncthreads();

    if (tid < num_experts) {
      prefix1[tid] = scan_buf1[tid];
    }
    if (tid <= num_experts) {
      cumsum1[tid] = prefix1[tid];
    }
  }

  // Phase 3: Compute cumsum for block_size2 (reuses shared_counts)
  int32_t padded_count2 = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count2 = (count + block_size2 - 1) / block_size2 * block_size2;
    scan_buf2[tid] = padded_count2;
  }

  {
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int num_warps_for_scan = (scan_size + WARP_SIZE - 1) / WARP_SIZE;
    const int warp_sum = warp_exclusive_scan(padded_count2) + padded_count2;
    if (lane_id == WARP_SIZE - 1) {
      warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
      int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
      int incl = warp_exclusive_scan(val) + val;
      warp_sums[tid] = incl;
    }
    __syncthreads();

    if (tid == 0) {
      prefix2[num_experts] = warp_sums[num_warps_for_scan - 1];
      s_total_tokens_post_pad2 = prefix2[num_experts];
      *total_tokens_post_pad2 = s_total_tokens_post_pad2;
    }
    __syncthreads();

    if (tid >= num_experts && tid < scan_size) {
      scan_buf2[tid] = 0;
    }
    __syncthreads();

    int v = (tid < scan_size) ? scan_buf2[tid] : 0;
    int pre = warp_exclusive_scan(v);
    if (lane_id == WARP_SIZE - 1) {
      warp_sums[warp_id] = pre + v;
    }
    __syncthreads();

    if (warp_id == 0) {
      int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
      warp_sums[lane_id] = warp_exclusive_scan(val);
    }
    __syncthreads();

    int offset = warp_sums[warp_id];
    if (tid < scan_size) {
      scan_buf2[tid] = pre + offset;
    }
    __syncthreads();

    if (tid < num_experts) {
      prefix2[tid] = scan_buf2[tid];
    }
    if (tid <= num_experts) {
      cumsum2[tid] = prefix2[tid];
    }
  }

  // Phase 4: Fill expert_ids for both block_m values
  {
    const int32_t num_blocks1 = s_total_tokens_post_pad1 / block_size1;
    for (int32_t i = tid; i < num_blocks1; i += stride) {
      int32_t block_start = i * block_size1;
      int left = 0, right = num_experts;
      while (left < right) {
        int mid = (left + right) >> 1;
        if (prefix1[mid] <= block_start) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }
      expert_ids1[i] = left - 1;
    }
  }
  {
    const int32_t num_blocks2 = s_total_tokens_post_pad2 / block_size2;
    for (int32_t i = tid; i < num_blocks2; i += stride) {
      int32_t block_start = i * block_size2;
      int left = 0, right = num_experts;
      while (left < right) {
        int mid = (left + right) >> 1;
        if (prefix2[mid] <= block_start) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }
      expert_ids2[i] = left - 1;
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename scalar_t>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel(FP8_TYPE* __restrict__ out, float* __restrict__ scale,
                                                          scalar_t const* __restrict__ input,
                                                          float const* __restrict__ scale_ub, const int hidden_size) {
  float const min_scaling_factor = 1.0f / (FP8_E4M3_MAX * 512.f);

  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;

  // Use int64 to avoid overflowing an int32 when calculating this offset
  int64_t offset = static_cast<int64_t>(token_idx) * hidden_size;
  scalar_t const* __restrict__ token_input = &input[offset];
  FP8_TYPE* __restrict__ token_output = &out[offset];

  float absmax_val = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float const x = static_cast<float>(token_input[i]);
    absmax_val = max(absmax_val, fabs(x));
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe = BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float token_scale;
  if (tid == 0) {
    if (scale_ub) {
      token_scale = min(block_absmax_val_maybe, *scale_ub);
    } else {
      token_scale = block_absmax_val_maybe;
    }
    // token scale computation
    token_scale = max(token_scale / FP8_E4M3_MAX, min_scaling_factor);
    scale[token_idx] = token_scale;
  }
  __syncthreads();

  // Note that we don't use inverted scales so we can match FBGemm impl.
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    token_output[i] = scaled_fp8_conversion<false>(static_cast<float>(token_input[i]), token_scale);
  }
}

template <typename scalar_t>
__global__ void segmented_max_reduction(float* __restrict__ scale, const scalar_t* __restrict__ input,
                                        int64_t num_elems) {
  __shared__ float cache[1024];
  int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

  // First store maximum for all values processes by
  // the current thread in cache[threadIdx.x]
  scalar_t tmp = 0.0;
  while (i < num_elems) {
    float x = static_cast<float>(input[i]);
    tmp = max(tmp, fabs(x));
    i += blockDim.x * gridDim.x;
  }
  cache[threadIdx.x] = tmp;

  __syncthreads();

  // Now perform parallel reduction within the thread block
  int ib = blockDim.x / 2;
  while (ib != 0) {
    if (threadIdx.x < ib && cache[threadIdx.x + ib] > cache[threadIdx.x]) {
      cache[threadIdx.x] = cache[threadIdx.x + ib];
    }
    __syncthreads();
    ib /= 2;
  }
  // Finally, since cache[0] contains the maximum for this thread block,
  // atomically write the max to the target location
  if (threadIdx.x == 0) {
    atomicMaxFloat(scale, cache[0] / FP8_E4M3_MAX);
  }
}

template <typename scalar_t, bool is_scale_inverted>
__device__ void scaled_fp8_conversion_vec(FP8_TYPE* __restrict__ out, scalar_t const* __restrict__ input,
                                          float const scale, int64_t const num_elems, int const tid, int const step) {
  using float8x4_t = q8x4_t<FP8_TYPE>;
  // Vectorized input/output to better utilize memory bandwidth.
  auto const* vectorized_in = reinterpret_cast<vec4_t<scalar_t> const*>(input);
  auto* vectorized_out = reinterpret_cast<float8x4_t*>(out);

  int64_t const num_vec_elems = num_elems >> 2;

#pragma unroll 4
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
    float8x4_t out_vec;

    out_vec.x = scaled_fp8_conversion<is_scale_inverted>(static_cast<float>(in_vec.x), scale);
    out_vec.y = scaled_fp8_conversion<is_scale_inverted>(static_cast<float>(in_vec.y), scale);
    out_vec.z = scaled_fp8_conversion<is_scale_inverted>(static_cast<float>(in_vec.z), scale);
    out_vec.w = scaled_fp8_conversion<is_scale_inverted>(static_cast<float>(in_vec.w), scale);
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    out[i] = scaled_fp8_conversion<is_scale_inverted>(static_cast<float>(input[i]), scale);
  }
}

template <typename scalar_t>
__global__ void scaled_fp8_quant_kernel(FP8_TYPE* __restrict__ out, const scalar_t* __restrict__ input,
                                        const float* __restrict__ scale, int64_t num_elems) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // Invert the scale so that we can use multiplications to avoid expensive
  // division.
  const float inverted_scale = 1.0f / (*scale);
  scaled_fp8_conversion_vec<scalar_t, true>(out, input, inverted_scale, num_elems, tid, blockDim.x * gridDim.x);
}

template <typename scalar_t, int TOPK, bool UseExpertParallel>
__global__ void moe_sum_kernel(scalar_t* __restrict__ out,          // [..., d]
                               const scalar_t* __restrict__ input,  // [..., topk, d]
                               const int* __restrict__ topk_ids,    // [..., topk], only used when UseExpertParallel
                               const int d) {
  const int64_t token_idx = blockIdx.x;

  // 128-bit load/store: half/bf16 -> 8 elements, float -> 4 elements
  constexpr int VEC_SIZE = 16 / sizeof(scalar_t);

  const int64_t d_vec = d / VEC_SIZE;
  const int64_t d_tail = d_vec * VEC_SIZE;

  const scalar_t* token_input = input + token_idx * TOPK * d;
  scalar_t* token_output = out + token_idx * d;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  // Vectorized main loop: each iteration loads/stores 16 bytes
  for (int64_t vec_idx = threadIdx.x; vec_idx < d_vec; vec_idx += blockDim.x) {
    vec_t<scalar_t, VEC_SIZE> acc;  // TODO(yfnjin): Maybe use float accumulation for better accuracy
    acc.fill(static_cast<scalar_t>(0.0f));

#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      if constexpr (UseExpertParallel) {
        if (topk_ids[token_idx * TOPK + k] < 0) {
          continue;
        }
      }
      vec_t<scalar_t, VEC_SIZE> v;
      v.load(token_input + k * d + vec_idx * VEC_SIZE);
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        acc[i] += v[i];
      }
    }

    // Store directly
    acc.store(token_output + vec_idx * VEC_SIZE);
  }

  // Scalar tail for remaining elements (when d % VEC_SIZE != 0)
  for (int64_t idx = d_tail + threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t acc = static_cast<scalar_t>(0.0f);
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      if constexpr (UseExpertParallel) {
        if (topk_ids[token_idx * TOPK + k] < 0) {
          continue;
        }
      }
      acc += token_input[k * d + idx];
    }
    token_output[idx] = acc;
  }
}

template <typename scalar_t>
__global__ void sum_out_dim1_kernel(scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
                                    const int num_tokens, const int topk, const int hidden_size) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < num_tokens && col < hidden_size) {
    scalar_t sum = 0.0f;
    for (int k = 0; k < topk; ++k) {
      sum += input[row * topk * hidden_size + k * hidden_size + col];
    }
    out[row * hidden_size + col] = sum;
  }
}

template <typename T, bool HAS_SWIGLU_LIMIT>
__global__ void silu_and_mul_kernel(const T* x, T* out, size_t size, size_t inter_size, float swiglu_limit) {
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= size) return;
  for (int64_t idx = threadIdx.x; idx < inter_size; idx += blockDim.x) {
    float a = *(&x[token_idx * 2 * inter_size + idx]);
    float b = *(&x[token_idx * 2 * inter_size + inter_size + idx]);
    if constexpr (HAS_SWIGLU_LIMIT) {
      a = fminf(a, swiglu_limit);
      b = fminf(fmaxf(b, -swiglu_limit), swiglu_limit);
    }
    out[token_idx * inter_size + idx] = a / (1.0f + expf(-a)) * b;
  }
}

template <typename T, bool HAS_SWIGLU_LIMIT>
__global__ void silu_and_mul_kernel_expert_parallel(const T* x, T* out, const int* topk_ids, size_t size,
                                                    size_t inter_size, float swiglu_limit) {
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= size) return;
  if (topk_ids[token_idx] < 0) return;
  for (int64_t idx = threadIdx.x; idx < inter_size; idx += blockDim.x) {
    float a = *(&x[token_idx * 2 * inter_size + idx]);
    float b = *(&x[token_idx * 2 * inter_size + inter_size + idx]);
    if constexpr (HAS_SWIGLU_LIMIT) {
      a = fminf(a, swiglu_limit);
      b = fminf(fmaxf(b, -swiglu_limit), swiglu_limit);
    }
    out[token_idx * inter_size + idx] = a / (1.0f + expf(-a)) * b;
  }
}

__device__ float fp8e4m3_to_float(uint8_t fp8) {
  uint8_t sign = (fp8 >> 7) & 0x1;
  uint8_t exponent = (fp8 >> 3) & 0xF;
  uint8_t mantissa = fp8 & 0x7;

  // 计算符号
  float sign_value = sign ? -1.0f : 1.0f;

  // 计算指数值
  int exponent_value = exponent - 7;  // FP8E4M3的指数偏移量为7

  // 计算尾数值
  float mantissa_value = 1.0f + (mantissa / 8.0f);

  // 组合符号、指数和尾数得到最终的float值
  // printf("%f * %f * 2^ %d\n", sign_value, mantissa_value, exponent_value);
  float result = sign_value * mantissa_value * std::pow(2.0f, exponent_value);

  return result;
}

template <typename T>
__global__ void weight_dequant_kernel(const uint8_t* data, const float* scale, T* output, int M, int N,
                                      int BLOCK_SIZE) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < static_cast<size_t>(M) * N) {
    size_t m = idx / N;
    size_t n = idx % N;
    float s = scale[m / BLOCK_SIZE * N / BLOCK_SIZE + n / BLOCK_SIZE];
    float w = fp8e4m3_to_float(data[idx]);
    float result = w * s;
    if constexpr (std::is_same<T, half>::value) {
      output[idx] = __float2half(result);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      output[idx] = __float2bfloat16(result);
    }
  }
}

template <typename T, typename TOKEN_CNTS_T, bool UseExpertParallel>
void InvokeMoeAlignBlockSize(T* topk_ids, int32_t* sorted_token_ids, int32_t* experts_ids,
                             int32_t* total_tokens_post_pad, const int32_t topk, const int32_t num_experts,
                             const int32_t expert_para_size, const int32_t block_size, const size_t numel,
                             const int32_t rank, const cudaStream_t& stream) {
  static int device_count = llm_kernels::utils::GetDeviceCount();
  static std::vector<bool> initialized(device_count, false);
  static std::vector<int> num_threads(device_count, 0);
  static std::vector<int> shared_mems(device_count, 0);
  static std::vector<int> num_experts_list(device_count, 0);
  static std::mutex init_mutex;
  if (!initialized[rank] || num_experts_list[rank] != num_experts) {
    std::lock_guard<std::mutex> lock(init_mutex);
    if (!initialized[rank] || num_experts_list[rank] != num_experts) {
      cudaSetDevice(rank);

      int device_max_shared_mem = llm_kernels::utils::getMaxSharedMemoryPerBlockOptin();
      int device_max_threads_per_block = llm_kernels::utils::getMaxThreadPerBlock();

      // 为该rank设置内核属性
      cudaError_t err = cudaFuncSetAttribute(moe_align_block_size_kernel_expert_parallel<T, TOKEN_CNTS_T>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, device_max_shared_mem);
      if (err != cudaSuccess) {
        printf(
            "CUDA error in InvokeMoeAlignBlockSize: cudaFuncSetAttribute(moe_align_block_size_kernel_expert_parallel) "
            "error in rank %d - %s\n",
            rank, cudaGetErrorString(err));
      }

      err = cudaFuncSetAttribute(moe_align_block_size_kernel<T, TOKEN_CNTS_T>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, device_max_shared_mem);
      if (err != cudaSuccess) {
        printf(
            "CUDA error in InvokeMoeAlignBlockSize: cudaFuncSetAttribute(moe_align_block_size_kernel) error in rank %d "
            "- %s\n",
            rank, cudaGetErrorString(err));
      }

      int target_num_thread = (device_max_shared_mem - (num_experts * expert_para_size + 1) * sizeof(int32_t)) /
                                  (num_experts * expert_para_size * sizeof(uint16_t)) -
                              2;
      target_num_thread = std::min(target_num_thread, device_max_threads_per_block);

      num_threads[rank] = target_num_thread;
      shared_mems[rank] = device_max_shared_mem;
      num_experts_list[rank] = num_experts;
      initialized[rank] = true;
    }
  }

  int num_thread = num_threads[rank];
  int shared_mem = shared_mems[rank];
  if constexpr (UseExpertParallel) {
    moe_align_block_size_kernel_expert_parallel<T, TOKEN_CNTS_T><<<1, num_thread, shared_mem, stream>>>(
        topk_ids, sorted_token_ids, experts_ids, total_tokens_post_pad, topk, num_experts, block_size, numel);
  } else {
    moe_align_block_size_kernel<T, TOKEN_CNTS_T><<<1, num_thread, shared_mem, stream>>>(
        topk_ids, sorted_token_ids, experts_ids, total_tokens_post_pad, num_experts, block_size, numel);
  }
}
#define INVOKE_MOE_ALIGN_BLOCK_SIZE(T, TOKEN_CNTS_T)                                                           \
  template void InvokeMoeAlignBlockSize<T, TOKEN_CNTS_T, true>(                                                \
      T * topk_ids, int32_t * sorted_token_ids, int32_t * experts_ids, int32_t * total_tokens_post_pad,        \
      const int32_t topk, const int32_t expert_para_size, const int32_t num_experts, const int32_t block_size, \
      const size_t numel, const int32_t rank, const cudaStream_t& stream);                                     \
  template void InvokeMoeAlignBlockSize<T, TOKEN_CNTS_T, false>(                                               \
      T * topk_ids, int32_t * sorted_token_ids, int32_t * experts_ids, int32_t * total_tokens_post_pad,        \
      const int32_t topk, const int32_t expert_para_size, const int32_t num_experts, const int32_t block_size, \
      const size_t numel, const int32_t rank, const cudaStream_t& stream);
INVOKE_MOE_ALIGN_BLOCK_SIZE(int32_t, uint16_t);
INVOKE_MOE_ALIGN_BLOCK_SIZE(int32_t, int32_t);
#undef INVOKE_MOE_ALIGN_BLOCK_SIZE

// `num_experts` should be greater than 64
template <typename T, bool UseExpertParallel>
void InvokeSglMoeAlignBlockSize(T* topk_ids, int32_t* sorted_token_ids, int32_t* expert_ids,
                                int32_t* total_tokens_post_pad, const int32_t max_num_tokens_padded,
                                const int32_t num_experts, const int32_t block_size, const size_t numel,
                                int32_t* cumsum, const cudaStream_t& stream) {
  const int max_blocks = 65535;

  const int align_block_threads = 1024;
  // We employ extra blocks to perform `sorted_ids.fill_(numel)`
  const int align_num_blocks =
      std::min(max_blocks, 1 + CEILDIV(max_num_tokens_padded / SGL_MOE_ALIGN_BLOCK_VEC_SIZE, align_block_threads));
  const size_t scan_size = llm_kernels::utils::NextPow2(num_experts);
  const size_t shared_mem_size = (num_experts + (num_experts + 1) + scan_size + WARP_SIZE) * sizeof(int32_t);

  auto* sgl_align_kernel = sgl_moe_align_block_size_kernel<T, UseExpertParallel>;
  LAUNCH_KERNEL_WITH_PDL(sgl_align_kernel, align_num_blocks, align_block_threads, shared_mem_size, stream, topk_ids,
                         sorted_token_ids, expert_ids, total_tokens_post_pad, max_num_tokens_padded, num_experts,
                         block_size, numel, cumsum, static_cast<int32_t>(scan_size));

  const int block_threads = 256;
  const int num_blocks = (numel + block_threads - 1) / block_threads;
  const int actual_blocks = std::min(num_blocks, max_blocks);

  auto* sgl_sort_kernel = sgl_moe_token_sort_kernel<T, UseExpertParallel>;
  LAUNCH_KERNEL_WITH_PDL(sgl_sort_kernel, actual_blocks, block_threads, 0, stream, topk_ids, sorted_token_ids, cumsum,
                         numel);
}
#define INVOKE_SGL_MOE_ALIGN_BLOCK_SIZE(T)                                                                          \
  template void InvokeSglMoeAlignBlockSize<T, true>(                                                                \
      T * topk_ids, int32_t * sorted_token_ids, int32_t * expert_ids, int32_t * total_tokens_post_pad,              \
      const int32_t max_num_tokens_padded, const int32_t num_experts, const int32_t block_size, const size_t numel, \
      int32_t* cumsum, const cudaStream_t& stream);                                                                 \
  template void InvokeSglMoeAlignBlockSize<T, false>(                                                               \
      T * topk_ids, int32_t * sorted_token_ids, int32_t * expert_ids, int32_t * total_tokens_post_pad,              \
      const int32_t max_num_tokens_padded, const int32_t num_experts, const int32_t block_size, const size_t numel, \
      int32_t* cumsum, const cudaStream_t& stream)
INVOKE_SGL_MOE_ALIGN_BLOCK_SIZE(int32_t);
#undef INVOKE_SGL_MOE_ALIGN_BLOCK_SIZE

// Merged align: compute alignment for two block_m values in one launch.
// Saves one topk_ids read + one kernel launch when GEMM1/GEMM2 have different block_m.
template <typename T, bool UseExpertParallel>
void InvokeSglMoeAlignBlockSizeMerged(T* topk_ids, int32_t* sorted_token_ids1, int32_t* expert_ids1,
                                      int32_t* total_tokens_post_pad1, int32_t* sorted_token_ids2, int32_t* expert_ids2,
                                      int32_t* total_tokens_post_pad2, const int32_t max_num_tokens_padded1,
                                      const int32_t max_num_tokens_padded2, const int32_t num_experts,
                                      const int32_t block_size1, const int32_t block_size2, const size_t numel,
                                      int32_t* cumsum1, int32_t* cumsum2, const cudaStream_t& stream) {
  const int max_blocks = 65535;
  const int align_block_threads = 1024;
  const int max_padded = std::max(max_num_tokens_padded1, max_num_tokens_padded2);
  const int align_num_blocks =
      std::min(max_blocks, 1 + CEILDIV(max_padded / SGL_MOE_ALIGN_BLOCK_VEC_SIZE, align_block_threads));
  const size_t scan_size = llm_kernels::utils::NextPow2(num_experts);
  // shared: counts[num_experts] + prefix1[num_experts+1] + scan1[scan_size] +
  //         prefix2[num_experts+1] + scan2[scan_size] + warp_sums[32]
  const size_t shared_mem_size =
      (num_experts + (num_experts + 1) + scan_size + (num_experts + 1) + scan_size + WARP_SIZE) * sizeof(int32_t);
  auto* sgl_merged_kernel = sgl_moe_align_block_size_merged_kernel<T, UseExpertParallel>;
  LAUNCH_KERNEL_WITH_PDL(sgl_merged_kernel, align_num_blocks, align_block_threads, shared_mem_size, stream, topk_ids,
                         sorted_token_ids1, expert_ids1, total_tokens_post_pad1, sorted_token_ids2, expert_ids2,
                         total_tokens_post_pad2, max_num_tokens_padded1, max_num_tokens_padded2, num_experts,
                         block_size1, block_size2, numel, cumsum1, cumsum2, scan_size);

  // Token sort: still need two separate launches (different cumsum buffers)
  const int block_threads = 256;
  const int num_blocks = (numel + block_threads - 1) / block_threads;
  const int actual_blocks = std::min(num_blocks, max_blocks);

  auto* sgl_sort_kernel = sgl_moe_token_sort_kernel<T, UseExpertParallel>;
  LAUNCH_KERNEL_WITH_PDL(sgl_sort_kernel, actual_blocks, block_threads, 0, stream, topk_ids, sorted_token_ids1, cumsum1,
                         numel);
  LAUNCH_KERNEL_WITH_PDL(sgl_sort_kernel, actual_blocks, block_threads, 0, stream, topk_ids, sorted_token_ids2, cumsum2,
                         numel);
}
#define INVOKE_SGL_MOE_ALIGN_BLOCK_SIZE_MERGED(T)                                                                   \
  template void InvokeSglMoeAlignBlockSizeMerged<T, true>(                                                          \
      T * topk_ids, int32_t * sorted_token_ids1, int32_t * expert_ids1, int32_t * total_tokens_post_pad1,           \
      int32_t * sorted_token_ids2, int32_t * expert_ids2, int32_t * total_tokens_post_pad2,                         \
      const int32_t max_num_tokens_padded1, const int32_t max_num_tokens_padded2, const int32_t num_experts,        \
      const int32_t block_size1, const int32_t block_size2, const size_t numel, int32_t* cumsum1, int32_t* cumsum2, \
      const cudaStream_t& stream);                                                                                  \
  template void InvokeSglMoeAlignBlockSizeMerged<T, false>(                                                         \
      T * topk_ids, int32_t * sorted_token_ids1, int32_t * expert_ids1, int32_t * total_tokens_post_pad1,           \
      int32_t * sorted_token_ids2, int32_t * expert_ids2, int32_t * total_tokens_post_pad2,                         \
      const int32_t max_num_tokens_padded1, const int32_t max_num_tokens_padded2, const int32_t num_experts,        \
      const int32_t block_size1, const int32_t block_size2, const size_t numel, int32_t* cumsum1, int32_t* cumsum2, \
      const cudaStream_t& stream)
INVOKE_SGL_MOE_ALIGN_BLOCK_SIZE_MERGED(int32_t);
#undef INVOKE_SGL_MOE_ALIGN_BLOCK_SIZE_MERGED

template <typename T>
void InvokeDynamicPerTokenScaledFP8Quant(const T* input, const float* scale, const void* scale_ub, void* output,
                                         const int hidden_size, const int num_tokens, const cudaStream_t& stream) {
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));
  dynamic_per_token_scaled_fp8_quant_kernel<T>
      <<<grid, block, 0, stream>>>(static_cast<FP8_TYPE*>(output), const_cast<float*>(scale), input,
                                   static_cast<float const*>(scale_ub), hidden_size);
}

template <typename T>
void InvokeDynamicScaledFP8Quant(const T* input, float* scale, void* output, const int num_tokens, const int num_elems,
                                 const cudaStream_t& stream) {
  dim3 grid(num_tokens);
  dim3 block(1024);
  segmented_max_reduction<T><<<grid, block, 0, stream>>>(scale, input, num_elems);
  scaled_fp8_quant_kernel<T><<<grid, block, 0, stream>>>(static_cast<FP8_TYPE*>(output), input, scale, num_elems);
}

template <typename T>
void InvokeStaticScaledFP8Quant(const T* input, float* scale, void* output, const int num_tokens, const int num_elems,
                                const cudaStream_t& stream) {
  dim3 grid(num_tokens);
  dim3 block(1024);
  scaled_fp8_quant_kernel<T><<<grid, block, 0, stream>>>(static_cast<FP8_TYPE*>(output), input, scale, num_elems);
}

template <typename T, bool UseExpertParallel>
void InvokeMoeSum(void* input,     // [num_tokens, topk, hidden_size]
                  void* output,    // [num_tokens, hidden_size]
                  void* topk_ids,  // [..., topk]
                  int num_tokens, int topk, int hidden_size, const cudaStream_t& stream) {
  dim3 grid(num_tokens);
  // token<=64: 1024 threads for best single-token latency.
  // token>64:  512 threads for better occupancy at scale.
  dim3 block((num_tokens <= 64) ? 1024 : 512);
  switch (topk) {
#define CASE_LAUNCH_MOE_SUM_KERNEL(TOPK)                                                                            \
  case TOPK: {                                                                                                      \
    constexpr auto kernel = moe_sum_kernel<T, TOPK, UseExpertParallel>;                                             \
    LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, reinterpret_cast<T*>(output),                            \
                           reinterpret_cast<const T*>(input), reinterpret_cast<const int*>(topk_ids), hidden_size); \
    break;                                                                                                          \
  }
    CASE_LAUNCH_MOE_SUM_KERNEL(1);
    CASE_LAUNCH_MOE_SUM_KERNEL(2);
    CASE_LAUNCH_MOE_SUM_KERNEL(3);
    CASE_LAUNCH_MOE_SUM_KERNEL(4);
    CASE_LAUNCH_MOE_SUM_KERNEL(5);
    CASE_LAUNCH_MOE_SUM_KERNEL(6);
    CASE_LAUNCH_MOE_SUM_KERNEL(7);
    CASE_LAUNCH_MOE_SUM_KERNEL(8);
    CASE_LAUNCH_MOE_SUM_KERNEL(9);
    CASE_LAUNCH_MOE_SUM_KERNEL(10);
    CASE_LAUNCH_MOE_SUM_KERNEL(11);
    CASE_LAUNCH_MOE_SUM_KERNEL(12);
    CASE_LAUNCH_MOE_SUM_KERNEL(13);
    CASE_LAUNCH_MOE_SUM_KERNEL(14);
    CASE_LAUNCH_MOE_SUM_KERNEL(15);
    CASE_LAUNCH_MOE_SUM_KERNEL(16);
    CASE_LAUNCH_MOE_SUM_KERNEL(32);
    CASE_LAUNCH_MOE_SUM_KERNEL(64);
#undef CASE_LAUNCH_MOE_SUM_KERNEL
    default:
      dim3 block_size(16, 16);
      dim3 grid_size((num_tokens + block_size.x - 1) / block_size.x, (hidden_size + block_size.y - 1) / block_size.y);

      sum_out_dim1_kernel<T><<<grid_size, block_size>>>(reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input),
                                                        num_tokens, topk, hidden_size);
      break;
  }
}
#define INVOKE_MOE_SUM(T)                                                                                   \
  template void InvokeMoeSum<T, true>(void* input, void* output, void* topk_ids, int num_tokens, int topk,  \
                                      int hidden_size, const cudaStream_t& stream);                         \
  template void InvokeMoeSum<T, false>(void* input, void* output, void* topk_ids, int num_tokens, int topk, \
                                       int hidden_size, const cudaStream_t& stream)
INVOKE_MOE_SUM(float);
INVOKE_MOE_SUM(half);
INVOKE_MOE_SUM(__nv_bfloat16);
#undef INVOKE_MOE_SUM

// ─── MoeFusedMulSum: fused multiply-by-topk-weight + sum across topk ────────
// Combines multiply-by-topk-weight + sum into a single kernel.
// Eliminates one kernel launch + topk * hidden_size * sizeof(T) bytes of
// redundant global memory traffic per token.
//
// input:  [num_tokens * topk, hidden_size] — GEMM2 output (read-only)
// topk_weights: [num_tokens * topk] (float) — flat layout, one weight per row
// topk_ids: [num_tokens, topk] (int) — for EP masking (skip if < 0)
// output: [num_tokens, hidden_size] — weighted sum across topk experts
template <typename T, int TOPK, bool UseExpertParallel>
__global__ void moe_fused_mul_sum_kernel(const T* __restrict__ input, const float* __restrict__ topk_weights,
                                         const int* __restrict__ topk_ids, T* __restrict__ output,
                                         const int hidden_size) {
  const int64_t token_idx = blockIdx.x;

  // 128-bit vectorized access: 8 elements for half/bf16, 4 for float
  constexpr int VEC_SIZE = 16 / sizeof(T);

  const int64_t d = hidden_size;
  const int64_t d_vec = d / VEC_SIZE;
  const int64_t d_tail = d_vec * VEC_SIZE;

  const T* token_input = input + token_idx * TOPK * d;
  T* token_output = output + token_idx * d;

  // Load topk_weights for this token (TOPK values)
  float weights[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    weights[k] = topk_weights[token_idx * TOPK + k];
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  // Vectorized main loop
  for (int64_t vec_idx = threadIdx.x; vec_idx < d_vec; vec_idx += blockDim.x) {
    // Use float accumulator for precision
    float acc[VEC_SIZE];
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      acc[i] = 0.0f;
    }

#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      if constexpr (UseExpertParallel) {
        if (topk_ids[token_idx * TOPK + k] < 0) {
          continue;
        }
      }
      vec_t<T, VEC_SIZE> v;
      v.load(token_input + k * d + vec_idx * VEC_SIZE);
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        acc[i] += static_cast<float>(v[i]) * weights[k];
      }
    }

    // Cast back to T and store
    vec_t<T, VEC_SIZE> out_vec;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      out_vec[i] = static_cast<T>(acc[i]);
    }
    out_vec.store(token_output + vec_idx * VEC_SIZE);
  }

  // Scalar tail for remaining elements
  for (int64_t idx = d_tail + threadIdx.x; idx < d; idx += blockDim.x) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      if constexpr (UseExpertParallel) {
        if (topk_ids[token_idx * TOPK + k] < 0) {
          continue;
        }
      }
      acc += static_cast<float>(token_input[k * d + idx]) * weights[k];
    }
    token_output[idx] = static_cast<T>(acc);
  }
}

template <typename T, bool UseExpertParallel>
void InvokeMoeFusedMulSum(const void* input,          // [num_tokens * topk, hidden_size]
                          const float* topk_weights,  // [num_tokens * topk] (flat)
                          const void* topk_ids,       // [num_tokens, topk]
                          void* output,               // [num_tokens, hidden_size]
                          int num_tokens, int topk, int hidden_size, const cudaStream_t& stream) {
  if (num_tokens == 0) return;

  dim3 grid(num_tokens);
  dim3 block((num_tokens <= 64) ? 1024 : 512);

  switch (topk) {
#define CASE_LAUNCH_MOE_FUSED_MUL_SUM(TOPK)                                                                    \
  case TOPK: {                                                                                                 \
    constexpr auto kernel = moe_fused_mul_sum_kernel<T, TOPK, UseExpertParallel>;                              \
    LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, reinterpret_cast<const T*>(input), topk_weights,    \
                           reinterpret_cast<const int*>(topk_ids), reinterpret_cast<T*>(output), hidden_size); \
    break;                                                                                                     \
  }
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(1);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(2);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(3);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(4);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(5);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(6);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(7);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(8);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(9);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(10);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(11);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(12);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(13);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(14);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(15);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(16);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(32);
    CASE_LAUNCH_MOE_FUSED_MUL_SUM(64);
#undef CASE_LAUNCH_MOE_FUSED_MUL_SUM
    default:
      KLLM_KERNEL_CHECK_WITH_INFO(false, "MoeFusedMulSum: unsupported topk value");
      break;
  }
}

#define INVOKE_MOE_FUSED_MUL_SUM(T)                                                                          \
  template void InvokeMoeFusedMulSum<T, true>(const void*, const float*, const void*, void*, int, int, int,  \
                                              const cudaStream_t&);                                          \
  template void InvokeMoeFusedMulSum<T, false>(const void*, const float*, const void*, void*, int, int, int, \
                                               const cudaStream_t&)
INVOKE_MOE_FUSED_MUL_SUM(float);
INVOKE_MOE_FUSED_MUL_SUM(half);
INVOKE_MOE_FUSED_MUL_SUM(__nv_bfloat16);
#undef INVOKE_MOE_FUSED_MUL_SUM

template <typename T, bool UseExpertParallel, bool HAS_SWIGLU_LIMIT>
void SiluAndMul(const T* input, T* output, const int* topk_ids, size_t elements_num, size_t inter_size,
                float swiglu_limit, const cudaStream_t& stream) {
  // input: dtype: FP16/BFP16, shape: [num_tokens, topk, inter_size * 2]
  // output: dtype: FP16/BFP16, shape: [num_tokens, topk, inter_size]
  // elements_num = num_tokens * topk * inter_size
  size_t total_expert_count = elements_num / 2 / inter_size;
  dim3 grid(total_expert_count);
  dim3 block(std::min(inter_size, static_cast<size_t>(1024)));
  if constexpr (UseExpertParallel) {
    silu_and_mul_kernel_expert_parallel<T, HAS_SWIGLU_LIMIT>
        <<<grid, block, 0, stream>>>(input, output, topk_ids, total_expert_count, inter_size, swiglu_limit);
  } else {
    silu_and_mul_kernel<T, HAS_SWIGLU_LIMIT>
        <<<grid, block, 0, stream>>>(input, output, total_expert_count, inter_size, swiglu_limit);
  }
}

#define SILU_AND_MUL(T)                                                                                           \
  template void SiluAndMul<T, true, true>(const T*, T*, const int*, size_t, size_t, float, const cudaStream_t&);  \
  template void SiluAndMul<T, true, false>(const T*, T*, const int*, size_t, size_t, float, const cudaStream_t&); \
  template void SiluAndMul<T, false, true>(const T*, T*, const int*, size_t, size_t, float, const cudaStream_t&); \
  template void SiluAndMul<T, false, false>(const T*, T*, const int*, size_t, size_t, float, const cudaStream_t&)
SILU_AND_MUL(float);
SILU_AND_MUL(half);
SILU_AND_MUL(__nv_bfloat16);
#undef SILU_AND_MUL

__device__ __forceinline__ float silu(const float& val) { return val / (1.0f + __expf(-val)); }

template <typename T>
void FlashinferSiluAndMul(const T* input, T* output, const int* topk_ids, size_t elements_num, size_t inter_size,
                          const cudaStream_t& stream) {
  size_t num_tokens = elements_num / 2 / inter_size;
  uint32_t vec_size = 16 / sizeof(T);
  uint32_t block_dim = std::min(inter_size / vec_size, static_cast<size_t>(1024));

  constexpr auto kernel = flashinfer::activation::act_and_mul_kernel<T, silu>;
  LAUNCH_KERNEL_WITH_PDL(kernel, static_cast<uint32_t>(num_tokens), block_dim, 0, stream, output, input,
                         static_cast<int>(inter_size));
}

#define FLASHINFER_SILU_AND_MUL(T)                                                                           \
  template void FlashinferSiluAndMul<T>(const T* input, T* output, const int* topk_ids, size_t elements_num, \
                                        size_t inter_size, const cudaStream_t& stream)
FLASHINFER_SILU_AND_MUL(float);
FLASHINFER_SILU_AND_MUL(half);
FLASHINFER_SILU_AND_MUL(__nv_bfloat16);
#undef FLASHINFER_SILU_AND_MUL

template <typename T, bool UseExpertParallel, bool HAS_SWIGLU_LIMIT>
void InvokeSiluAndMul(const T* input, T* output, const int* topk_ids, size_t elements_num, size_t inter_size,
                      float swiglu_limit, const cudaStream_t& stream) {
  if constexpr (UseExpertParallel) {
    SiluAndMul<T, true, HAS_SWIGLU_LIMIT>(input, output, topk_ids, elements_num, inter_size, swiglu_limit, stream);
  } else {
    if constexpr (HAS_SWIGLU_LIMIT) {
      // flashinfer 的 act_and_mul_kernel 不支持 clamp, 强制走本地 kernel
      SiluAndMul<T, false, true>(input, output, topk_ids, elements_num, inter_size, swiglu_limit, stream);
    } else {
#if defined(ENABLE_FLASHINFER)
      FlashinferSiluAndMul<T>(input, output, topk_ids, elements_num, inter_size, stream);
#else
      SiluAndMul<T, false, false>(input, output, topk_ids, elements_num, inter_size, 0.0f, stream);
#endif
    }
  }
}

#define INVOKE_SILU_AND_MUL(T)                                                                                         \
  template void InvokeSiluAndMul<T, true, true>(const T*, T*, const int*, size_t, size_t, float, const cudaStream_t&); \
  template void InvokeSiluAndMul<T, true, false>(const T*, T*, const int*, size_t, size_t, float,                      \
                                                 const cudaStream_t&);                                                 \
  template void InvokeSiluAndMul<T, false, true>(const T*, T*, const int*, size_t, size_t, float,                      \
                                                 const cudaStream_t&);                                                 \
  template void InvokeSiluAndMul<T, false, false>(const T*, T*, const int*, size_t, size_t, float, const cudaStream_t&)
INVOKE_SILU_AND_MUL(float);
INVOKE_SILU_AND_MUL(half);
INVOKE_SILU_AND_MUL(__nv_bfloat16);
#undef INVOKE_SILU_AND_MUL

template <typename T>
void InvokeWeightDequant(const uint8_t* x, const float* s, T* output, int M, int N, int block_size,
                         const cudaStream_t& stream) {
  dim3 block(256);
  dim3 grid((M * N + 256 - 1) / 256);
  weight_dequant_kernel<T><<<grid, block, 0, stream>>>(x, s, output, M, N, block_size);
}

#define INVOKE_WEIGHT_DEQUANT(T)                                                                                  \
  template void InvokeWeightDequant<T>(const uint8_t* x, const float* s, T* output, int M, int N, int block_size, \
                                       const cudaStream_t& stream)
INVOKE_WEIGHT_DEQUANT(half);
INVOKE_WEIGHT_DEQUANT(float);
INVOKE_WEIGHT_DEQUANT(__nv_bfloat16);
#undef INVOKE_WEIGHT_DEQUANT

__global__ void fill_buffer_kernel(int* buffer, const int* fill_info, int fill_info_length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int32_t i = 0; i < fill_info_length; i += 3) {
    int32_t start_offset = fill_info[i];
    int32_t length = fill_info[i + 1];
    if (idx >= start_offset && idx < start_offset + length) {
      buffer[idx] = fill_info[i + 2];
      break;
    }
  }
}

void InvokeFillIntToBuffer(int* output, void* fill_info, int* fill_info_on_host, int fill_info_length,
                           const cudaStream_t& stream) {
  cudaMemcpyAsync(fill_info, reinterpret_cast<void*>(fill_info_on_host), fill_info_length * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  // [start_offset0, length0, fill_value0, ... , start_offsetN, lengthN, fill_valueN]
  const int total_elems = fill_info_on_host[fill_info_length - 2] + fill_info_on_host[fill_info_length - 3];

  const int BLOCK_SIZE = 256;
  int num_blocks = ceil(static_cast<float>(total_elems) / BLOCK_SIZE);
  fill_buffer_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, static_cast<int*>(fill_info), fill_info_length);
}

// FP8 dynamic quantization — needed for humming W4A8 MoE path
template void InvokeDynamicScaledFP8Quant<__nv_bfloat16>(const __nv_bfloat16* input, float* scale, void* output,
                                                         const int num_tokens, const int num_elems,
                                                         const cudaStream_t& stream);
template void InvokeDynamicScaledFP8Quant<__half>(const __half* input, float* scale, void* output, const int num_tokens,
                                                  const int num_elems, const cudaStream_t& stream);
template void InvokeDynamicScaledFP8Quant<float>(const float* input, float* scale, void* output, const int num_tokens,
                                                 const int num_elems, const cudaStream_t& stream);

// FP8 per-token dynamic quantization — correct function for humming W4A8 (per-token scale)
template void InvokeDynamicPerTokenScaledFP8Quant<__nv_bfloat16>(const __nv_bfloat16* input, const float* scale,
                                                                 const void* scale_ub, void* output,
                                                                 const int hidden_size, const int num_tokens,
                                                                 const cudaStream_t& stream);
template void InvokeDynamicPerTokenScaledFP8Quant<__half>(const __half* input, const float* scale, const void* scale_ub,
                                                          void* output, const int hidden_size, const int num_tokens,
                                                          const cudaStream_t& stream);
template void InvokeDynamicPerTokenScaledFP8Quant<float>(const float* input, const float* scale, const void* scale_ub,
                                                         void* output, const int hidden_size, const int num_tokens,
                                                         const cudaStream_t& stream);

// 每个 thread 读一个 topk_ids[i]，把对应专家位置的 mask byte 置 1。
// 同一 byte 上不同 thread 写入相同的常量值 1，按 PTX 内存模型语义不构成竞争——
// 即便底层是 read-modify-write，最终值仍然是 1。
template <typename T>
__global__ void build_active_expert_mask_kernel(const T* __restrict__ topk_ids,
                                                uint8_t* __restrict__ active_expert_mask, int32_t numel,
                                                int32_t num_experts) {
  const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t stride = gridDim.x * blockDim.x;
  for (int32_t i = tid; i < numel; i += stride) {
    const int32_t eid = static_cast<int32_t>(topk_ids[i]);
    if (eid >= 0 && eid < num_experts) {
      active_expert_mask[eid] = 1;
    }
  }
}

template <typename T>
void InvokeBuildActiveExpertMask(const T* topk_ids, uint8_t* active_expert_mask, int32_t numel, int32_t num_experts,
                                 const cudaStream_t& stream) {
  if (active_expert_mask == nullptr || num_experts <= 0) {
    return;
  }
  // 1) 整 mask 清零；num_experts 通常 ≤ 256，bytes 极小，cudaMemsetAsync 几乎免费
  cudaMemsetAsync(active_expert_mask, 0, static_cast<size_t>(num_experts), stream);

  // 2) scatter：标记命中专家。numel 极少超过 chunk_size(32K) * top_k，启动开销 < 几 us
  if (numel <= 0) {
    return;
  }
  constexpr int kBlockSize = 256;
  const int num_blocks = static_cast<int>((static_cast<int64_t>(numel) + kBlockSize - 1) / kBlockSize);
  build_active_expert_mask_kernel<T>
      <<<num_blocks, kBlockSize, 0, stream>>>(topk_ids, active_expert_mask, numel, num_experts);
}

#define INSTANTIATE_BUILD_ACTIVE_EXPERT_MASK(T)                                                               \
  template void InvokeBuildActiveExpertMask<T>(const T* topk_ids, uint8_t* active_expert_mask, int32_t numel, \
                                               int32_t num_experts, const cudaStream_t& stream)
INSTANTIATE_BUILD_ACTIVE_EXPERT_MASK(int32_t);
INSTANTIATE_BUILD_ACTIVE_EXPERT_MASK(uint32_t);
#undef INSTANTIATE_BUILD_ACTIVE_EXPERT_MASK
}  // namespace nvidia
}  // namespace llm_kernels
