/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope.cuh
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

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/forward_mode.h"
#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/plan.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/tile.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/type.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/utils.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/vec.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/warp.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

/// \brief common block size for memory-bound kernel
constexpr uint32_t kFNRBlockSize = 128;
constexpr uint32_t kFNRNumWarps = kFNRBlockSize / device::kWarpThreads;

struct FusedNormRopeParams {
  void* __restrict__ input;
  const void* __restrict__ weight;
  float eps;
  uint32_t num_works;
  const void* __restrict__ handle;
  const void* __restrict__ freqs_cis;
  uint32_t compress_ratio;
};

// One warp per work item.
// Part 1: RMSNorm with weight, store ALL elements (including rope region) back.
// Part 2: Apply RoPE to the last kRopeDim elements.
//
// kMode is a compile-time enum dispatched by the host launcher.
template <typename DType, int64_t kHeadDim, int64_t kRopeDim, ForwardMode kMode, bool kUsePDL>
__global__ void fused_norm_rope_kernel(const __grid_constant__ FusedNormRopeParams params) {
  using namespace device;

  constexpr int64_t kMaxVecSize = 16 / sizeof(DType);
  constexpr int64_t kVecSize = std::min(kMaxVecSize, kHeadDim / kWarpThreads);
  constexpr int64_t kLocalSize = kHeadDim / (kWarpThreads * kVecSize);
  constexpr int64_t kRopeVecSize = kRopeDim / (kWarpThreads * 2);
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  static_assert(kHeadDim % (kWarpThreads * kVecSize) == 0);
  static_assert(kLocalSize * kVecSize * kWarpThreads == kHeadDim);
  static_assert(kRopeDim % (kWarpThreads * 2) == 0);
  static_assert(kRopeDim % (kVecSize * kLocalSize) == 0);
  static_assert(kRopeSize <= kWarpThreads);
  static_assert(kRopeVecSize == 1, "only support rope dim = 64");

  const auto& [_input, _weight, eps, num_works,   // norm
               handle, freqs_cis, compress_ratio  // rope
  ] = params;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kFNRNumWarps + warp_id;

  if (work_id >= num_works) [[unlikely]]
    return;

  // Wait here because reading the plan depends on the previous kernel's write.
  PDLWaitPrimary<kUsePDL>();

  DType* input;
  int32_t position;
  if constexpr (kMode == ForwardMode::CompressExtend) {
    const auto plan = static_cast<const PrefillPlan*>(handle)[work_id];
    if (plan.ragged_id == 0xFFFFFFFF) [[unlikely]]
      return;
    input = static_cast<DType*>(_input) + plan.ragged_id * kHeadDim;
    position = plan.position + 1 - compress_ratio;
  } else if constexpr (kMode == ForwardMode::CompressDecode) {
    input = static_cast<DType*>(_input) + work_id * kHeadDim;
    const auto seq_len = static_cast<const int32_t*>(handle)[work_id];
    if (seq_len % static_cast<int32_t>(compress_ratio) != 0) return;
    position = seq_len - static_cast<int32_t>(compress_ratio);
  } else if constexpr (kMode == ForwardMode::DefaultForward) {
    input = static_cast<DType*>(_input) + work_id * kHeadDim;
    position = static_cast<const int64_t*>(handle)[work_id];
  }

  using Storage = AlignedVector<DType, kVecSize>;
  __shared__ Storage s_rope_input[kFNRNumWarps][kRopeSize];

  // prefetch freq (split layout: [cos_0..cos_{N/2-1}, sin_0..sin_{N/2-1}], stored as DType)
  const auto mem_freq = tile::Memory<DType>::warp();
  const DType* freq_base = static_cast<const DType*>(freqs_cis) + position * kRopeDim;
  const auto freq_cos_raw = mem_freq.load(freq_base);
  const auto freq_sin_raw = mem_freq.load(freq_base + kRopeDim / 2);
  const float freq_cos = cast<float>(freq_cos_raw);
  const float freq_sin = cast<float>(freq_sin_raw);

  // PDLWaitPrimary<kUsePDL>();

  // part 1: norm
  {
    const auto gmem = tile::Memory<Storage>::warp();
    Storage input_vec[kLocalSize];
    Storage weight_vec[kLocalSize];
#pragma unroll
    for (int i = 0; i < kLocalSize; ++i) {
      input_vec[i] = gmem.load(input, i);
    }

#pragma unroll
    for (int i = 0; i < kLocalSize; ++i) {
      weight_vec[i] = gmem.load(_weight, i);
    }

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        const auto fp32_input = cast<float>(input_vec[i][j]);
        sum_of_squares += fp32_input * fp32_input;
      }
    }

    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + eps);

#pragma unroll
    for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        const auto fp32_input = cast<float>(input_vec[i][j]);
        const auto fp32_weight = cast<float>(weight_vec[i][j]);
        input_vec[i][j] = cast<DType>(fp32_input * norm_factor * fp32_weight);
      }
    }

    const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;

#pragma unroll
    for (int i = 0; i < kLocalSize; ++i) {
      if (i == kLocalSize - 1 && is_rope_lane) {
        const auto rope_id = lane_id - (kWarpThreads - kRopeSize);
        s_rope_input[warp_id][rope_id] = input_vec[i];
      } else {
        gmem.store(input, input_vec[i], i);
      }
    }

    __syncwarp();
  }

  // part 2: rope
  {
    // mem elem = DType x 2
    using DTypex2_t = packed_t<DType>;
    const auto mem_elem = tile::Memory<DTypex2_t>::warp();
    const auto elem = mem_elem.load(s_rope_input[warp_id]);
    const auto [x_real, x_imag] = cast<fp32x2_t>(elem);
    const fp32x2_t output = {
        x_real * freq_cos - x_imag * freq_sin,
        x_real * freq_sin + x_imag * freq_cos,
    };
    mem_elem.store(input + (kHeadDim - kRopeDim), cast<DTypex2_t>(output));
  }

  PDLTriggerSecondary<kUsePDL>();
}

// In-place RMSNorm followed by RoPE on the last kRopeDim elements.
//
// `input`  : [num_q_tokens, kHeadDim] DType, modified in-place.
// `weight` : [kHeadDim] DType, RMSNorm scale.
// `handle` : depends on mode –
//              CompressExtend : PrefillPlan* [num_works]
//              CompressDecode : int32_t*    [num_works]  (seq_lens)
//              DefaultForward : int64_t*    [num_works]  (positions)
// `freqs_cis` : [-1, kRopeDim] DType (split layout: first half cos, second half sin)
// `num_works`  : number of active work items (warps).
// `compress_ratio` : must be > 0 for Compress modes, 0 for DefaultForward.
template <typename DType, int64_t kHeadDim, int64_t kRopeDim = 64>
inline void compress_fused_norm_rope_inplace(void* input, const void* weight, const void* handle, const void* freqs_cis,
                                             uint32_t num_works, ForwardMode mode, float eps, uint32_t compress_ratio,
                                             cudaStream_t stream) {
  if (num_works == 0) return;

  const uint32_t num_blocks = (num_works + kFNRNumWarps - 1u) / kFNRNumWarps;
  const FusedNormRopeParams params{input, weight, eps, num_works, handle, freqs_cis, compress_ratio};

  switch (mode) {
    case ForwardMode::CompressExtend: {
      constexpr auto kernel =
          fused_norm_rope_kernel<DType, kHeadDim, kRopeDim, ForwardMode::CompressExtend, /*kUsePDL=*/true>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kFNRBlockSize, 0, stream, params);
      break;
    }
    case ForwardMode::CompressDecode: {
      constexpr auto kernel =
          fused_norm_rope_kernel<DType, kHeadDim, kRopeDim, ForwardMode::CompressDecode, /*kUsePDL=*/true>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kFNRBlockSize, 0, stream, params);
      break;
    }
    case ForwardMode::DefaultForward: {
      constexpr auto kernel =
          fused_norm_rope_kernel<DType, kHeadDim, kRopeDim, ForwardMode::DefaultForward, /*kUsePDL=*/true>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kFNRBlockSize, 0, stream, params);
      break;
    }
  }
}

// InFloat / OutFloat = __nv_bfloat16 or float; kHeadDim = 128 or 512.
#define INSTANTIATE_FNR(DType, kHeadDim)                           \
  template void compress_fused_norm_rope_inplace<DType, kHeadDim>( \
      void*, const void*, const void*, const void*, uint32_t, ForwardMode, float, uint32_t, cudaStream_t);

INSTANTIATE_FNR(__nv_bfloat16, 128)
INSTANTIATE_FNR(__nv_bfloat16, 512)
INSTANTIATE_FNR(float, 128)
INSTANTIATE_FNR(float, 512)

#undef INSTANTIATE_FNR

}  // namespace nvidia
}  // namespace llm_kernels
