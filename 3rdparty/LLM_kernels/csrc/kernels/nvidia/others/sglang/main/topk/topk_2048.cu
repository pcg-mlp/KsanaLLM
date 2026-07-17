/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/elementwise/topk.cu
 *
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
 */

#include "csrc/kernels/nvidia/others/sglang/main/topk/topk.h"

#include <cuda_fp16.h>

#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/pdl_compat.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {
namespace topk {

// Helpers specific to the K=2048 variant are kept inside a sub-namespace.
namespace k2048 {

// ---------------------------------------------------------------------------
// Upstream body starts here. Modifications w.r.t. the original:
//   * `C10_LIKELY` -> `[[likely]]` (no torch dep).
//   * The torch-tensor-based interface (`fast_topk_transform_interface`) is
//     replaced by a plain C++ launcher that takes raw device pointers and
//     follows the Ksana page-table convention shared with the K=512/K=1024
//     variants (page_bits, raw_indices, batch_indices, extend_len).
//   * Loops over `kTopK` because `kTopK (2048) > kTopKBlockSize (1024)`.
// ---------------------------------------------------------------------------

constexpr uint32_t kTopK = 2048;
constexpr uint32_t kTopKBlockSize = 1024;
// Reduced from 128KB to 32KB to improve occupancy.
// Each radix pass needs at most ~kTopK candidates in the threshold bin,
// so 4K entries per round (2 rounds = 8K entries = 32KB) is sufficient.
constexpr uint32_t kSMEM = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)

struct TopK2048Params {
  const float* __restrict__ scores;
  const int32_t* __restrict__ row_starts;
  const int32_t* __restrict__ row_ends;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;  // optional: output raw abs position indices before page transform
  const int64_t score_stride;
  const int64_t page_table_stride;
  uint32_t page_bits;
  // For batch_id computation:
  //   Prefill (kIsPrefill=true):  batch_indices[work_id] = batch_id (direct O(1) lookup).
  //   Decode  (kIsPrefill=false): extend_len = works per batch; batch_id = work_id / extend_len.
  const int32_t* __restrict__ batch_indices;  // [num_works], used in prefill
  uint32_t extend_len;                        // scalar, used in decode
};

__device__ __forceinline__ uint8_t convert_to_uint8(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ int32_t page_to_indices(const int32_t* __restrict__ page_table, uint32_t i,
                                                   uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

[[maybe_unused]] __device__ __forceinline__ void naive_transform(
    const float* __restrict__,  // unused
    const int32_t* __restrict__ page_table, int32_t* __restrict__ indices,
    int32_t* __restrict__ raw_indices,  // optional: output raw abs position indices
    const uint32_t length, const uint32_t page_bits) {
  static_assert(kTopK % kTopKBlockSize == 0);
  const auto tx = threadIdx.x;
#pragma unroll
  for (uint32_t i = tx; i < kTopK; i += kTopKBlockSize) {
    if (i < length) {
      indices[i] = page_to_indices(page_table, i, page_bits);
      if (raw_indices != nullptr) {
        raw_indices[i] = static_cast<int32_t>(i);
      }
    } else {
      indices[i] = -1;  // fill invalid indices to -1
      if (raw_indices != nullptr) {
        raw_indices[i] = -1;
      }
    }
  }
}

[[maybe_unused]] __device__ __forceinline__ void radix_topk(const float* __restrict__ input,
                                                            int32_t* __restrict__ output, const uint32_t length) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t BLOCK_SIZE = kTopKBlockSize;
  constexpr uint32_t SMEM_INPUT_SIZE = kSMEM / (2 * sizeof(int32_t));

  alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ uint32_t s_counter;
  alignas(128) __shared__ uint32_t s_threshold_bin_id;
  alignas(128) __shared__ uint32_t s_num_input[2];
  alignas(128) __shared__ int32_t s_last_remain;

  extern __shared__ uint32_t s_input_idx[][kSMEM / (2 * sizeof(int32_t))];

  const uint32_t tx = threadIdx.x;
  uint32_t remain_topk = kTopK;
  auto& s_histogram = _s_histogram_buf[0];

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int32_t i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = _s_histogram_buf[k][tx];
        if (tx + j < RADIX) {
          value += _s_histogram_buf[k][tx + j];
        }
        _s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();
  for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();
  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  remain_topk -= s_histogram[threshold_bin + 1];
  if (remain_topk == 0) {
    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
      const uint32_t bin = convert_to_uint8(input[idx]);
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
      const float raw_input = input[idx];
      const uint32_t bin = convert_to_uint8(raw_input);
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (pos < SMEM_INPUT_SIZE) {
          [[likely]] s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto raw_num_input = s_num_input[r_idx];
    const auto num_input = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = remain_topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    remain_topk -= s_histogram[threshold_bin + 1];

    if (remain_topk == 0) {
      for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              output[kTopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (pos < SMEM_INPUT_SIZE) {
              /// NOTE: (dark) fuse the histogram computation here
              [[likely]] s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

template <bool kUsePDL, bool kIsPrefill>
__global__ __launch_bounds__(kTopKBlockSize) void topk_2048_transform(const __grid_constant__ TopK2048Params params) {
  const auto& [scores, row_starts, row_ends, page_table, page_indices, raw_indices,  // pointers
               score_stride, page_table_stride, page_bits,                           // sizes
               batch_indices, extend_len                                             // batch dispatch
  ] = params;
  const uint32_t work_id = blockIdx.x;

  // Compute batch_id from work_id
  uint32_t batch_id;
  if constexpr (kIsPrefill) {
    // Prefill: direct O(1) lookup from precomputed batch_indices.
    batch_id = static_cast<uint32_t>(batch_indices[work_id]);
  } else {
    // Decode: uniform work count per batch.
    batch_id = work_id / extend_len;
  }

  /// NOTE: dangerous prefetch row range before PDL wait
  const int32_t start = row_starts[work_id];
  const int32_t end = row_ends[work_id];
  const uint32_t seq_len = (end > start) ? static_cast<uint32_t>(end - start) : 0u;
  const auto score_ptr = scores + work_id * score_stride + start;
  const auto page_ptr = page_table + batch_id * page_table_stride;
  const auto indices_ptr = page_indices + work_id * kTopK;
  const auto raw_indices_ptr = raw_indices != nullptr ? raw_indices + work_id * kTopK : nullptr;

  PDLWaitPrimary<kUsePDL>();

  if (seq_len <= kTopK) {
    naive_transform(score_ptr, page_ptr, indices_ptr, raw_indices_ptr, seq_len, page_bits);
  } else {
    __shared__ int32_t s_topk_indices[kTopK];
    radix_topk(score_ptr, s_topk_indices, seq_len);
    static_assert(kTopK % kTopKBlockSize == 0);
    const auto tx = threadIdx.x;
#pragma unroll
    for (uint32_t i = tx; i < kTopK; i += kTopKBlockSize) {
      indices_ptr[i] = page_to_indices(page_ptr, s_topk_indices[i], page_bits);
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[i] = s_topk_indices[i];
      }
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <auto* f, size_t kMaxDynamicSMEM>
inline void setup_kernel_smem_once() {
  [[maybe_unused]] static const auto result = [] {
    return ::cudaFuncSetAttribute(reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(kMaxDynamicSMEM));
  }();
  (void)result;
}

}  // namespace k2048

// ---------------------------------------------------------------------------
// Host-side launcher (Ksana-local rewrite of SGLang's fast_topk_transform_interface)
//
// Prefill: pass batch_indices (device array [num_works]) and set extend_len=0.
// Decode:  pass batch_indices=nullptr and set extend_len to the uniform work count per batch.
// ---------------------------------------------------------------------------
void launch_topk2048(const float* scores, const int32_t* row_starts, const int32_t* row_ends, const int32_t* page_table,
                     int32_t* page_indices_out, int32_t* raw_indices_out, int64_t score_stride,
                     int64_t page_table_stride, uint32_t num_works, uint32_t page_bits, const int32_t* batch_indices,
                     uint32_t extend_len, cudaStream_t stream) {
  const k2048::TopK2048Params params{
      /*.scores=*/scores,
      /*.row_starts=*/row_starts,
      /*.row_ends=*/row_ends,
      /*.page_table=*/page_table,
      /*.page_indices=*/page_indices_out,
      /*.raw_indices=*/raw_indices_out,
      /*.score_stride=*/score_stride,
      /*.page_table_stride=*/page_table_stride,
      /*.page_bits=*/page_bits,
      /*.batch_indices=*/batch_indices,
      /*.extend_len=*/extend_len,
  };
  constexpr auto kSMEM_ = k2048::kSMEM + sizeof(int32_t);  // align up a little

  if (batch_indices != nullptr) {
    // Prefill path: use precomputed batch_indices
    constexpr auto kernel = k2048::topk_2048_transform</*kUsePDL*/ true, /*kIsPrefill*/ true>;
    k2048::setup_kernel_smem_once<kernel, kSMEM_>();
    LAUNCH_KERNEL_WITH_PDL(kernel, num_works, k2048::kTopKBlockSize, kSMEM_, stream, params);
  } else {
    // Decode path
    constexpr auto kernel = k2048::topk_2048_transform</*kUsePDL*/ true, /*kIsPrefill*/ false>;
    k2048::setup_kernel_smem_once<kernel, kSMEM_>();
    LAUNCH_KERNEL_WITH_PDL(kernel, num_works, k2048::kTopKBlockSize, kSMEM_, stream, params);
  }
}

}  // namespace topk
}  // namespace nvidia
}  // namespace llm_kernels
