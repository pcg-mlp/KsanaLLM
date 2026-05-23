/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh
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

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/fused_rope_quant_scale.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/deepseek_v4/fp8_utils.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/math.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/type.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/utils.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/vec.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/warp.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {


// =============================================================================
// Constants
// =============================================================================
using DType = __nv_bfloat16;
static constexpr int64_t kHeadDim = 128;
static constexpr int64_t kRopeDim = 64;
static constexpr uint32_t kBlockSize = 128;
static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
static constexpr int64_t kVecSize = 4;
static constexpr uint32_t kRopeSize = kRopeDim / kVecSize;  // = 16

// =============================================================================
// Kernel parameters struct
// =============================================================================
struct FusedRopeQuantScaleParams {
  const void* __restrict__ q;          // [batch * heads, head_dim] bf16, contiguous
  const void* __restrict__ weights;    // [batch * heads] bf16
  const void* __restrict__ freqs_cis;  // [max_pos, rope_dim] bf16, split [cos|sin]
  const void* __restrict__ positions;  // [batch] int64
  void* __restrict__ output_q_fp8;     // [batch * heads, head_dim] fp8, contiguous
  float* __restrict__ output_scale;    // [batch * heads] f32
  float combined_scale;
  uint32_t num_heads;
  uint32_t batch_size;
};

// =============================================================================
// Kernel
// =============================================================================
// Each warp handles one (token, head).
//
// Template parameters (additions over SGLang):
//   kRopeLaneStart - the lane index where rope dims begin:
//     0  → rope occupies lanes [0,15]  (elements [0,63]),  i.e. [rope, nope] layout
//     16 → rope occupies lanes [16,31] (elements [64,127]), i.e. [nope, rope] layout
//   kIsNeox - true: NeoX style (split-half: first half = real, second half = imag)
//             false: GPT-J interleaved style (adjacent pairs: [real, imag, real, imag, ...])
template <bool kUsePDL, uint32_t kRopeLaneStart, bool kIsNeox>
__global__ __launch_bounds__(kBlockSize, 16) void fused_rope_quant_scale_kernel(
    const __grid_constant__ FusedRopeQuantScaleParams params) {
  using namespace device;
  using deepseek_v4::fp8::pack_fp8;
  using DTypex2_t = packed_t<DType>;

  constexpr uint32_t kRopeLaneEnd = kRopeLaneStart + kRopeSize;
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);
  static_assert(kRopeLaneEnd <= kWarpThreads);

  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;  // 4 fp8 / lane

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kNumWarps + warp_id;

  const bool is_rope_lane =
      (kRopeLaneStart == 0) ? (lane_id < kRopeLaneEnd) : (lane_id >= kRopeLaneStart && lane_id < kRopeLaneEnd);

  const uint32_t total_works = params.batch_size * params.num_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_heads;
  const auto input_ptr = static_cast<const DType*>(params.q) + work_id * kHeadDim;
  const auto position = static_cast<const int64_t*>(params.positions)[batch_id];
  const auto* freq_base = static_cast<const DType*>(params.freqs_cis) + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data;
  const auto weight_val = cast<float>(static_cast<const DType*>(params.weights)[work_id]);

  // ── Part 1: Load Q head. Each lane owns a 4-elem pack. ──
  {
    Storage input_vec;
    input_vec.load(input_ptr, lane_id);
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = cast<float>(input_vec[i]);
    }
  }

  // ── Part 2: RoPE on rope lanes only (4 elems / lane = 2 (real, imag) pairs). ──
  if constexpr (kIsNeox) {
    // NeoX style: within the rope region, first half = real, second half = imag.
    // All 32 lanes participate in shfl (full warp convergence); only rope lanes apply rotation.
    constexpr uint32_t half_rope_lanes = kRopeSize / 2;  // = 8
    const uint32_t rope_lane_offset = lane_id - kRopeLaneStart;
    const bool is_real_lane = rope_lane_offset < half_rope_lanes;
    const uint32_t pair_lane = is_real_lane ? (lane_id + half_rope_lanes) : (lane_id - half_rope_lanes);
    const uint32_t freq_idx = (rope_lane_offset % half_rope_lanes) * kVecSize;

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const float other = __shfl_sync(0xFFFFFFFFu, data[i], pair_lane);
      if (is_rope_lane) {
        const float f_cos = cast<float>(freq_base[freq_idx + i]);
        const float f_sin = cast<float>(freq_base[kRopeDim / 2 + freq_idx + i]);
        if (is_real_lane) {
          data[i] = data[i] * f_cos - other * f_sin;
        } else {
          data[i] = other * f_sin + data[i] * f_cos;
        }
      }
    }
  } else {
    // GPT-J interleaved style: 4 elems / lane = 2 (real, imag) pairs.
    if (is_rope_lane) {
      const uint32_t rope_lane_offset = lane_id - kRopeLaneStart;
      const uint32_t freq_idx = rope_lane_offset * 2;
      const float f_cos_0 = cast<float>(freq_base[freq_idx]);
      const float f_sin_0 = cast<float>(freq_base[kRopeDim / 2 + freq_idx]);
      const float f_cos_1 = cast<float>(freq_base[freq_idx + 1]);
      const float f_sin_1 = cast<float>(freq_base[kRopeDim / 2 + freq_idx + 1]);

      const auto x_real = data[0];
      const auto x_imag = data[1];
      const auto y_real = data[2];
      const auto y_imag = data[3];
      data[0] = x_real * f_cos_0 - x_imag * f_sin_0;
      data[1] = x_real * f_sin_0 + x_imag * f_cos_0;
      data[2] = y_real * f_cos_1 - y_imag * f_sin_1;
      data[3] = y_real * f_sin_1 + y_imag * f_cos_1;
    }
  }

  // NOTE(yfnjin): We skip the Hadamard (transform), since it has little impact on accuracy

  // ── Part 3: FP8 abs-max quantization + store ──
  {
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = math::max(local_max, math::abs(data[i]));
    }
    const auto abs_max = warp::reduce_max(local_max);
    const auto scale = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
    const auto inv_scale = 1.0f / scale;
    OutStorage result;
    result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
    result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);

    // q_fp8 row pointer: 128 fp8 / row = 32 OutStorage / row, one per lane.
    auto* out_row = static_cast<uint8_t*>(params.output_q_fp8) + work_id * kHeadDim;
    result.store(out_row, lane_id);
    if (lane_id == 0) {
      params.output_scale[work_id] = weight_val * params.combined_scale * scale;
    }
  }

  // We trigger here for safety
  PDLTriggerSecondary<kUsePDL>();
}

// =============================================================================
// Host launcher
// =============================================================================
void fused_rope_quant_scale(const void* q, const void* weights, const void* freqs_cis, const void* positions,
                            void* output_q_fp8, float* output_scale, float n_heads_inv_sqrt, float softmax_scale,
                            uint32_t num_heads, uint32_t batch_size, uint32_t rope_offset, cudaStream_t stream,
                            bool is_neox) {
  if (batch_size == 0) return;

  const uint32_t total_warps = batch_size * num_heads;
  const uint32_t num_blocks = device::div_ceil(total_warps, kNumWarps);

  const float combined_scale = n_heads_inv_sqrt * softmax_scale;
  const FusedRopeQuantScaleParams params{
      q, weights, freqs_cis, positions, output_q_fp8, output_scale, combined_scale, num_heads, batch_size};

  // Dispatch by rope_offset → compile-time rope_lane_start, and is_neox
  const uint32_t rope_lane_start = rope_offset / kVecSize;
  if (is_neox) {
    if (rope_lane_start == 0) {
      constexpr auto kernel = fused_rope_quant_scale_kernel</*kUsePDL=*/true, /*kRopeLaneStart=*/0, /*kIsNeox=*/true>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kBlockSize, 0, stream, params);
    } else {
      constexpr auto kernel = fused_rope_quant_scale_kernel</*kUsePDL=*/true, /*kRopeLaneStart=*/16, /*kIsNeox=*/true>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kBlockSize, 0, stream, params);
    }
  } else {
    if (rope_lane_start == 0) {
      constexpr auto kernel = fused_rope_quant_scale_kernel</*kUsePDL=*/true, /*kRopeLaneStart=*/0, /*kIsNeox=*/false>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kBlockSize, 0, stream, params);
    } else {
      constexpr auto kernel = fused_rope_quant_scale_kernel</*kUsePDL=*/true, /*kRopeLaneStart=*/16, /*kIsNeox=*/false>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kBlockSize, 0, stream, params);
    }
  }
}

}  // namespace nvidia
}  // namespace llm_kernels
