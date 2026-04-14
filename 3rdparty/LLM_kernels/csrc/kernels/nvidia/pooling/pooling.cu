/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
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
 */

#include "csrc/kernels/nvidia/pooling/pooling.h"

#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

// --------------------------------------------------------------------------
// Vectorized trait: maps element type T to a 128-bit vector type (16 bytes).
//   half          → 8 elements per vector load/store
//   __nv_bfloat16 → 8 elements per vector load/store
//   float         → 4 elements per vector load/store
// --------------------------------------------------------------------------
template <typename T>
struct VecType {
  using type = T;
  static constexpr int32_t kSize = 1;
};

template <>
struct VecType<half> {
  using type = float4;  // 16 bytes = 8 × sizeof(half)
  static constexpr int32_t kSize = 8;
};

template <>
struct VecType<__nv_bfloat16> {
  using type = float4;  // 16 bytes = 8 × sizeof(__nv_bfloat16)
  static constexpr int32_t kSize = 8;
};

template <>
struct VecType<float> {
  using type = float4;  // 16 bytes = 4 × sizeof(float)
  static constexpr int32_t kSize = 4;
};

// Choose BLOCK_SIZE based on the number of work items (vec_count or hidden_size).
// Goal: each thread handles 1–4 loads. Clamp to [64, 512] in powers of 2.
inline int32_t ChooseBlockSize(int32_t work_items) {
  if (work_items <= 64) return 64;
  if (work_items <= 256) return 128;
  if (work_items <= 1024) return 256;
  return 512;
}

// --------------------------------------------------------------------------
// [Optimization C] Improved warp + block reduction with __shfl_sync broadcast.
//
// Changes vs. original:
//   - After warp-0 finishes the cross-warp reduction, the result is broadcast
//     to ALL threads via __shfl_sync, eliminating the extra shared-memory
//     round-trip + __syncthreads that was previously used for broadcast.
//   - Returns the reduced sum in ALL threads (not just thread 0).
// --------------------------------------------------------------------------
__device__ __forceinline__ float BlockReduceSumAllThreads(float val) {
  // Step 1: intra-warp reduction via shuffle.
  for (int32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }

  // Step 2: inter-warp reduction — each warp's lane-0 writes its partial sum
  //         to shared memory, then warp 0 reduces those partials.
  __shared__ float shared_sums[32];  // max 32 warps (1024 threads)
  const int32_t lane_id = threadIdx.x & 31;
  const int32_t warp_id = threadIdx.x >> 5;

  if (lane_id == 0) {
    shared_sums[warp_id] = val;
  }
  __syncthreads();

  const int32_t num_warps = (blockDim.x + warpSize - 1) / warpSize;
  // Only warp 0 does the final cross-warp reduction.
  float result;
  if (warp_id == 0) {
    result = (lane_id < num_warps) ? shared_sums[lane_id] : 0.0f;
    for (int32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
      result += __shfl_down_sync(0xffffffff, result, offset);
    }
    // result is now correct only in lane 0 of warp 0.
    // The shared-memory broadcast below distributes it to all threads.
  }

  // Step 3: broadcast from warp 0 to all other warps.
  //   Write to shared memory once, then all warps read.
  if (warp_id == 0 && lane_id == 0) {
    shared_sums[0] = result;
  }
  __syncthreads();
  result = shared_sums[0];

  return result;
}

// ==========================================================================
// L2NormalizeKernel — Optimized with shared-memory caching (single-pass
// global memory read), rsqrtf, improved block reduction, and compile-time
// BLOCK_SIZE selection.
//
// Optimizations applied:
//   [A] Shared memory cache: Phase1 reads from global memory into shared
//       memory and computes sum-of-squares. Phase2 reads from shared memory
//       (not global memory) to normalize, then writes back. This cuts global
//       memory reads from 2× to 1× (total traffic: 1R + 1W instead of 2R + 1W).
//   [B] rsqrtf: replaces sqrtf + fdiv with a single SFU instruction.
//   [C] BlockReduceSumAllThreads: broadcasts the reduction result to all
//       threads via __shfl_sync, avoiding an extra __syncthreads barrier.
//   [D] BLOCK_SIZE is a template parameter chosen at launch time based on
//       hidden_size, so the compiler can fully unroll the inner loop.
//
// For the shared-memory path, each block needs hidden_size * sizeof(T) bytes
// of dynamic shared memory.  If hidden_size is too large to fit in shared
// memory, the kernel falls back to the original two-pass global memory path.
// ==========================================================================

// ---------------------------------------------------------------------------
// Kernel variant 1: Shared-memory cached, vectorized, single-pass global read.
// Requires: hidden_size % kVecSize == 0 AND hidden_size * sizeof(T) <= smem.
// ---------------------------------------------------------------------------
template <typename T, int BLOCK_SIZE>
__global__ void L2NormalizeKernelSmem(T* data, const int32_t hidden_size) {
  using Vec = typename VecType<T>::type;
  constexpr int32_t kVecSize = VecType<T>::kSize;

  const int32_t row = blockIdx.x;
  T* row_data = data + row * hidden_size;
  Vec* vec_data = reinterpret_cast<Vec*>(row_data);
  const int32_t vec_count = hidden_size / kVecSize;

  // Dynamic shared memory as a cache for the entire row.
  extern __shared__ __align__(16) char smem_raw[];
  Vec* smem_vec = reinterpret_cast<Vec*>(smem_raw);

  // ---- Phase 1: Load from global → shared memory, compute sum of squares ----
  float local_sum = 0.0f;
  for (int32_t i = threadIdx.x; i < vec_count; i += BLOCK_SIZE) {
    Vec v = vec_data[i];
    smem_vec[i] = v;  // cache in shared memory
    const T* elems = reinterpret_cast<const T*>(&v);
#pragma unroll
    for (int32_t j = 0; j < kVecSize; ++j) {
      float fval = static_cast<float>(elems[j]);
      local_sum += fval * fval;
    }
  }

  // Block-wide reduction — result is available in ALL threads.
  float sum_sq = BlockReduceSumAllThreads(local_sum);

  // [Optimization B] rsqrtf: single SFU instruction instead of sqrtf + fdiv.
  float inv_norm = (sum_sq > 0.0f) ? rsqrtf(sum_sq) : 1.0f;

  // ---- Phase 2: Read from shared memory, normalize, write to global ----
  // Note: __syncthreads() is NOT needed here because BlockReduceSumAllThreads
  // already ends with a __syncthreads() (line 99), which guarantees all Phase 1
  // shared memory writes are visible to all threads before we read them below.
  for (int32_t i = threadIdx.x; i < vec_count; i += BLOCK_SIZE) {
    Vec v = smem_vec[i];
    T* elems = reinterpret_cast<T*>(&v);
#pragma unroll
    for (int32_t j = 0; j < kVecSize; ++j) {
      float fval = static_cast<float>(elems[j]);
      elems[j] = static_cast<T>(fval * inv_norm);
    }
    vec_data[i] = v;
  }
}

// ---------------------------------------------------------------------------
// Kernel variant 2: Fallback — two-pass global memory (no shared memory cache).
// Handles arbitrary hidden_size including non-aligned tail elements.
// Still benefits from optimizations B, C, D.
// ---------------------------------------------------------------------------
template <typename T, int BLOCK_SIZE>
__global__ void L2NormalizeKernelGlobal(T* data, const int32_t hidden_size) {
  using Vec = typename VecType<T>::type;
  constexpr int32_t kVecSize = VecType<T>::kSize;

  const int32_t row = blockIdx.x;
  T* row_data = data + row * hidden_size;

  const bool use_vec = (hidden_size % kVecSize == 0);
  const int32_t vec_count = use_vec ? (hidden_size / kVecSize) : 0;
  const int32_t tail_start = vec_count * kVecSize;
  Vec* vec_data = reinterpret_cast<Vec*>(row_data);

  // ---- Phase 1: compute sum of squares ----
  float local_sum = 0.0f;
  if (use_vec) {
    for (int32_t i = threadIdx.x; i < vec_count; i += BLOCK_SIZE) {
      Vec v = vec_data[i];
      const T* elems = reinterpret_cast<const T*>(&v);
#pragma unroll
      for (int32_t j = 0; j < kVecSize; ++j) {
        float fval = static_cast<float>(elems[j]);
        local_sum += fval * fval;
      }
    }
  }
  for (int32_t i = tail_start + threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
    float fval = static_cast<float>(row_data[i]);
    local_sum += fval * fval;
  }

  // Block-wide reduction — result is available in ALL threads.
  float sum_sq = BlockReduceSumAllThreads(local_sum);

  // [Optimization B] rsqrtf.
  float inv_norm = (sum_sq > 0.0f) ? rsqrtf(sum_sq) : 1.0f;

  // ---- Phase 2: normalize in-place (re-read from global memory) ----
  if (use_vec) {
    for (int32_t i = threadIdx.x; i < vec_count; i += BLOCK_SIZE) {
      Vec v = vec_data[i];
      T* elems = reinterpret_cast<T*>(&v);
#pragma unroll
      for (int32_t j = 0; j < kVecSize; ++j) {
        float fval = static_cast<float>(elems[j]);
        elems[j] = static_cast<T>(fval * inv_norm);
      }
      vec_data[i] = v;
    }
  }
  for (int32_t i = tail_start + threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
    float fval = static_cast<float>(row_data[i]);
    row_data[i] = static_cast<T>(fval * inv_norm);
  }
}

// ==========================================================================
// Host dispatch: selects BLOCK_SIZE and kernel variant at runtime.
//
// [Optimization D] Dynamic BLOCK_SIZE selection based on hidden_size:
//   - hidden_size <=   128  → BLOCK_SIZE =  64  (avoid idle threads)
//   - hidden_size <=   512  → BLOCK_SIZE = 128
//   - hidden_size <= 2048   → BLOCK_SIZE = 256
//   - hidden_size >  2048   → BLOCK_SIZE = 512  (minimize per-thread iterations)
//
// [Optimization A] Shared-memory path is used when:
//   (1) hidden_size is a multiple of kVecSize (alignment), AND
//   (2) hidden_size * sizeof(T) fits in per-SM shared memory (≤ 48 KB default).
// ==========================================================================

// Maximum shared memory per block that we request (48 KB is the default limit
// on all CUDA architectures without opting in to extended shared memory).
static constexpr int32_t kMaxSmemBytes = 48 * 1024;

template <typename T>
void InvokeL2Normalize(T* data, const int32_t batch_size, const int32_t hidden_size, cudaStream_t stream) {
  if (data == nullptr || batch_size == 0 || hidden_size == 0) return;

  constexpr int32_t kVecSize = VecType<T>::kSize;
  const bool is_aligned = (hidden_size % kVecSize == 0);
  const int32_t smem_bytes = hidden_size * sizeof(T);
  const bool use_smem = is_aligned && (smem_bytes <= kMaxSmemBytes);

  // [Optimization D] Choose BLOCK_SIZE based on hidden_size.
  //   Goal: each thread should handle 1–4 vector loads (sweet spot).
  //   vec_count = hidden_size / kVecSize.
  //   We want BLOCK_SIZE ≈ vec_count / 2 (so each thread does ~2 loads).
  //   Clamp to [64, 512] in powers of 2.
  int32_t vec_count = is_aligned ? (hidden_size / kVecSize) : hidden_size;
  int32_t block_size = ChooseBlockSize(vec_count);

  dim3 grid(batch_size);

  // Macro to dispatch a specific BLOCK_SIZE at compile time.
#define LAUNCH_SMEM_KERNEL(BS)                                                                     \
  L2NormalizeKernelSmem<T, BS><<<grid, BS, smem_bytes, stream>>>(data, hidden_size)

#define LAUNCH_GLOBAL_KERNEL(BS)                                                                   \
  L2NormalizeKernelGlobal<T, BS><<<grid, BS, 0, stream>>>(data, hidden_size)

  if (use_smem) {
    switch (block_size) {
      case 64:  LAUNCH_SMEM_KERNEL(64);  break;
      case 128: LAUNCH_SMEM_KERNEL(128); break;
      case 256: LAUNCH_SMEM_KERNEL(256); break;
      case 512: LAUNCH_SMEM_KERNEL(512); break;
      default:  LAUNCH_SMEM_KERNEL(256); break;
    }
  } else {
    switch (block_size) {
      case 64:  LAUNCH_GLOBAL_KERNEL(64);  break;
      case 128: LAUNCH_GLOBAL_KERNEL(128); break;
      case 256: LAUNCH_GLOBAL_KERNEL(256); break;
      case 512: LAUNCH_GLOBAL_KERNEL(512); break;
      default:  LAUNCH_GLOBAL_KERNEL(256); break;
    }
  }

#undef LAUNCH_SMEM_KERNEL
#undef LAUNCH_GLOBAL_KERNEL

  CHECK_NVIDIA_CUDA_ERROR(cudaGetLastError());
}

template void InvokeL2Normalize(float* data, const int32_t batch_size, const int32_t hidden_size, cudaStream_t stream);
template void InvokeL2Normalize(half* data, const int32_t batch_size, const int32_t hidden_size, cudaStream_t stream);
template void InvokeL2Normalize(__nv_bfloat16* data, const int32_t batch_size, const int32_t hidden_size,
                                cudaStream_t stream);

// ==========================================================================
// MeanPoolingKernel — Average hidden states of each sequence.
//
// Grid:  (batch_size,)  — one block per sequence.
// Block: (BLOCK_SIZE,)  — threads collaborate to process hidden_size elements.
//
// For each sequence i (from offsets[i] to offsets[i+1]):
//   output[i, d] = mean(input[offsets[i]:offsets[i+1], d])
//
// Accumulation is done in float32 for numerical stability.
// Vectorized loads are used when hidden_size is a multiple of kVecSize.
// ==========================================================================

template <typename T, int BLOCK_SIZE>
__global__ void MeanPoolingKernel(const T* __restrict__ input, const size_t* __restrict__ offsets,
                                  const int32_t hidden_size, T* __restrict__ output) {
  using Vec = typename VecType<T>::type;
  constexpr int32_t kVecSize = VecType<T>::kSize;

  const int32_t seq_idx = blockIdx.x;
  const size_t seq_start = offsets[seq_idx];
  const size_t seq_end = offsets[seq_idx + 1];
  const int32_t seq_len = static_cast<int32_t>(seq_end - seq_start);

  T* out_row = output + seq_idx * hidden_size;

  if (seq_len <= 0) {
    // Empty sequence: write zeros.
    for (int32_t i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
      out_row[i] = static_cast<T>(0.0f);
    }
    return;
  }

  const float inv_len = 1.0f / static_cast<float>(seq_len);
  const bool use_vec = (hidden_size % kVecSize == 0);

  if (use_vec) {
    const int32_t vec_count = hidden_size / kVecSize;
    Vec* out_vec = reinterpret_cast<Vec*>(out_row);

    // Each thread processes a subset of vector positions across the hidden dimension.
    for (int32_t vi = threadIdx.x; vi < vec_count; vi += BLOCK_SIZE) {
      // Accumulate in float32.
      float accum[kVecSize];
#pragma unroll
      for (int32_t j = 0; j < kVecSize; ++j) {
        accum[j] = 0.0f;
      }

      // Sum across all tokens in this sequence.
      for (int32_t tok = 0; tok < seq_len; ++tok) {
        const T* tok_row = input + (seq_start + tok) * hidden_size;
        const Vec* tok_vec = reinterpret_cast<const Vec*>(tok_row);
        Vec v = tok_vec[vi];
        const T* elems = reinterpret_cast<const T*>(&v);
#pragma unroll
        for (int32_t j = 0; j < kVecSize; ++j) {
          accum[j] += static_cast<float>(elems[j]);
        }
      }

      // Compute mean and store.
      Vec result;
      T* result_elems = reinterpret_cast<T*>(&result);
#pragma unroll
      for (int32_t j = 0; j < kVecSize; ++j) {
        result_elems[j] = static_cast<T>(accum[j] * inv_len);
      }
      out_vec[vi] = result;
    }
  } else {
    // Scalar fallback for non-aligned hidden_size.
    for (int32_t d = threadIdx.x; d < hidden_size; d += BLOCK_SIZE) {
      float accum = 0.0f;
      for (int32_t tok = 0; tok < seq_len; ++tok) {
        accum += static_cast<float>(input[(seq_start + tok) * hidden_size + d]);
      }
      out_row[d] = static_cast<T>(accum * inv_len);
    }
  }
}

template <typename T>
void InvokeMeanPooling(const T* input, const size_t* offsets, const int32_t batch_size, const int32_t hidden_size,
                       T* output, cudaStream_t stream) {
  if (input == nullptr || offsets == nullptr || output == nullptr || batch_size == 0 || hidden_size == 0) return;

  constexpr int32_t kVecSize = VecType<T>::kSize;
  const bool is_aligned = (hidden_size % kVecSize == 0);

  // Choose BLOCK_SIZE similarly to L2Normalize.
  int32_t vec_count = is_aligned ? (hidden_size / kVecSize) : hidden_size;
  int32_t block_size = ChooseBlockSize(vec_count);

  dim3 grid(batch_size);

#define LAUNCH_MEAN_KERNEL(BS) \
  MeanPoolingKernel<T, BS><<<grid, BS, 0, stream>>>(input, offsets, hidden_size, output)

  switch (block_size) {
    case 64:  LAUNCH_MEAN_KERNEL(64);  break;
    case 128: LAUNCH_MEAN_KERNEL(128); break;
    case 256: LAUNCH_MEAN_KERNEL(256); break;
    case 512: LAUNCH_MEAN_KERNEL(512); break;
    default:  LAUNCH_MEAN_KERNEL(256); break;
  }

#undef LAUNCH_MEAN_KERNEL

  CHECK_NVIDIA_CUDA_ERROR(cudaGetLastError());
}

template void InvokeMeanPooling(const float* input, const size_t* offsets, const int32_t batch_size,
                                const int32_t hidden_size, float* output, cudaStream_t stream);
template void InvokeMeanPooling(const half* input, const size_t* offsets, const int32_t batch_size,
                                const int32_t hidden_size, half* output, cudaStream_t stream);
template void InvokeMeanPooling(const __nv_bfloat16* input, const size_t* offsets, const int32_t batch_size,
                                const int32_t hidden_size, __nv_bfloat16* output, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
