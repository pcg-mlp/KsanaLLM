/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/4807f6c5220c8ac53e395d6d2ed493cda241bc8e/python/sglang/jit_kernel/csrc/deepseek_v4/topk_1024.cuh
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

// Helpers specific to the K=1024 variant are kept inside a sub-namespace so
// that `topk.cuh` and `topk_1024.cuh` can be translated into the same library
// without symbol collisions. The upstream file placed them in an anonymous
// namespace; we preserve behaviour while making the split explicit.
namespace k1024 {

// ---------------------------------------------------------------------------
// Upstream body starts here. Modifications w.r.t. the original:
//   * `SGL_DEVICE` -> `__device__ __forceinline__`
//   * `TopK1024Kernel::transform(...)` (templated on tvm::ffi::TensorView) is
//     replaced by a plain C++ launcher that takes raw device pointers; the
//     kernel and all `__device__` helpers are kept byte-for-byte identical.
// ---------------------------------------------------------------------------

constexpr uint32_t kTopK = 1024;
constexpr uint32_t kTopKBlockSize = 1024;
constexpr uint32_t kSMEM = 16 * 1024 * sizeof(uint32_t);  // 64KB (bytes)

struct TopK1024Params {
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
  static_assert(kTopK <= kTopKBlockSize);
  if (const auto tx = threadIdx.x; tx < length) {
    indices[tx] = page_to_indices(page_table, tx, page_bits);
    if (raw_indices != nullptr) {
      raw_indices[tx] = tx;
    }
  } else if (kTopK == kTopKBlockSize || tx < kTopK) {
    indices[tx] = -1;  // fill invalid indices to -1
    if (raw_indices != nullptr) {
      raw_indices[tx] = -1;
    }
  }
}

[[maybe_unused]] __device__ __forceinline__ void radix_topk(const float* __restrict__ input,
                                                            int32_t* __restrict__ output, const uint32_t length) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t BLOCK_SIZE = kTopKBlockSize;
  constexpr uint32_t SMEM_INPUT_SIZE = kSMEM / (2 * sizeof(int32_t));

  alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 32];
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
__global__ void topk_1024_transform(const __grid_constant__ TopK1024Params params) {
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
    static_assert(kTopK <= kTopKBlockSize);
    const auto tx = threadIdx.x;
    if (kTopK == kTopKBlockSize || tx < kTopK) {
      indices_ptr[tx] = page_to_indices(page_ptr, s_topk_indices[tx], page_bits);
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[tx] = s_topk_indices[tx];
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

}  // namespace k1024

// ---------------------------------------------------------------------------
// Host-side launcher (Ksana-local rewrite of SGLang's TopK1024Kernel::transform)
//
// Prefill: pass batch_indices (device array [num_works]) and set extend_len=0.
// Decode:  pass batch_indices=nullptr and set extend_len to the uniform work count per batch.
// ---------------------------------------------------------------------------
void launch_topk1024(const float* scores, const int32_t* row_starts, const int32_t* row_ends, const int32_t* page_table,
                     int32_t* page_indices_out, int32_t* raw_indices_out, int64_t score_stride,
                     int64_t page_table_stride, uint32_t num_works, uint32_t page_bits, const int32_t* batch_indices,
                     uint32_t extend_len, cudaStream_t stream) {
  const k1024::TopK1024Params params{
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
  constexpr auto kSMEM_ = k1024::kSMEM + sizeof(int32_t);  // align up a little

  if (batch_indices != nullptr) {
    // Prefill path: use precomputed batch_indices
    constexpr auto kernel = k1024::topk_1024_transform</*kUsePDL*/ true, /*kIsPrefill*/ true>;
    k1024::setup_kernel_smem_once<kernel, kSMEM_>();
    LAUNCH_KERNEL_WITH_PDL(kernel, num_works, k1024::kTopKBlockSize, kSMEM_, stream, params);
  } else {
    // Decode path
    constexpr auto kernel = k1024::topk_1024_transform</*kUsePDL*/ true, /*kIsPrefill*/ false>;
    k1024::setup_kernel_smem_once<kernel, kSMEM_>();
    LAUNCH_KERNEL_WITH_PDL(kernel, num_works, k1024::kTopKBlockSize, kSMEM_, stream, params);
  }
}

}  // namespace topk
}  // namespace nvidia
}  // namespace llm_kernels
