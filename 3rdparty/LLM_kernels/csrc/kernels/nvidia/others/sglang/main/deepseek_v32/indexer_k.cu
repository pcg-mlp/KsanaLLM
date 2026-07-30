/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v32/indexer_k.cuh
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

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v32/indexer_k.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/deepseek_v4/fp8_utils.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/math.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/type.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/utils.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/vec.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/warp.cuh"
#include "csrc/utils/nvidia/assert.h"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

// =============================================================================
// Constants
// =============================================================================
static constexpr uint32_t kBlockSize = 128;
static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;

// Indexer cache layout constants:
//   128 FP8 bytes + 4 bytes float32 scale = 132 bytes total
// Page layout: [value_region: 128 * page_size] [scale_region: 4 * page_size]
constexpr int kIndexerTokenDataBytes = 128;  // 128 FP8 values
constexpr int kIndexerScaleBytes = 4;        // 1 float32 scale per token

#define K_INDEXER_KERNEL __global__ __launch_bounds__(kBlockSize, 16)

// =============================================================================
// Kernel parameters
// =============================================================================
struct FusedKIndexerNormRopeStoreParams {
  const void* __restrict__ k_input;    // (B, 128) bf16
  void* __restrict__ cache;            // (num_pages, page_stride) uint8
  const void* __restrict__ indices;    // (B,) int32 (store_out_loc; -1 skips store)
  const void* __restrict__ weight;     // (128,) bf16  -- LayerNorm gamma
  const void* __restrict__ bias;       // (128,) bf16  -- LayerNorm beta
  const void* __restrict__ freqs_cis;  // (max_pos, 64) bf16 split [cos|sin]
  const void* __restrict__ positions;  // (B,) int64
  int64_t k_input_stride_batch;
  int64_t page_stride;  // bytes between consecutive pages/blocks in cache
  uint32_t batch_size;
  float eps;
};

// =============================================================================
// Fused LayerNorm + RoPE + fp8 quant + paged store
// =============================================================================
template <bool kUsePDL, bool kIsNeox, int32_t kPageBits>
K_INDEXER_KERNEL void fused_k_indexer_norm_rope_store_kernel(
    const __grid_constant__ FusedKIndexerNormRopeStoreParams params) {
  using namespace device;
  using deepseek_v4::fp8::pack_fp8;
  using DType = __nv_bfloat16;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;  // = 16
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);

  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;  // 4 fp8 / lane

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kNumWarps + warp_id;
  const bool is_rope_lane = lane_id < kRopeSize;

  if (work_id >= params.batch_size) [[unlikely]]
    return;

  const auto index = static_cast<const int32_t*>(params.indices)[work_id];
  // Skip norm+rope+quantize entirely if index=-1
  if (index == -1) [[unlikely]] {
    return;
  }

  const auto input_ptr = static_cast<const DType*>(params.k_input) + work_id * params.k_input_stride_batch;
  const auto position = static_cast<const int64_t*>(params.positions)[work_id];
  const auto* freq_base = static_cast<const DType*>(params.freqs_cis) + position * kRopeDim;

  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  void* page_ptr = pointer::offset(params.cache, page * params.page_stride);
  void* value_ptr = pointer::offset(page_ptr, offset * kIndexerTokenDataBytes);
  void* scale_ptr = pointer::offset(page_ptr, kIndexerTokenDataBytes << kPageBits, offset * kIndexerScaleBytes);

  PDLWaitPrimary<kUsePDL>();
  Float4 data;

  // Part 1: LayerNorm (gamma/beta are bf16 in Ksana; cast to fp32 on load).
  {
    Storage input_vec, gamma_vec, beta_vec;
    input_vec.load(input_ptr, lane_id);
    gamma_vec.load(static_cast<const DType*>(params.weight), lane_id);
    beta_vec.load(static_cast<const DType*>(params.bias), lane_id);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = cast<float>(input_vec[i]);
      sum += data[i];
    }
    const float mean = warp::reduce_sum(sum) / kHeadDim;

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const float centered = data[i] - mean;
      var += centered * centered;
    }
    const float inv_std = math::rsqrt(warp::reduce_sum(var) / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const float g = cast<float>(gamma_vec[i]);
      const float b = cast<float>(beta_vec[i]);
      data[i] = (data[i] - mean) * inv_std * g + b;
    }
  }

  // Part 2: RoPE on rope lanes.
  // freqs_cis is bf16 (max_pos, kRopeDim) in split layout: first half cos, second half sin.
  if constexpr (kIsNeox) {
    // NeoX style: first half = real parts, second half = imag parts within the rope region.
    // All 32 lanes participate in shfl; only rope lanes apply rotation.
    constexpr uint32_t half_rope_lanes = kRopeSize / 2;  // = 8
    const uint32_t rope_lane_offset = lane_id;           // rope region starts at lane 0
    const bool is_real_lane = is_rope_lane && rope_lane_offset < half_rope_lanes;
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
      const uint32_t freq_idx = lane_id * 2;
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

  // Part 3: fp8 act-quant + paged store.
  const float local_max = fmaxf(fmaxf(fabsf(data[0]), fabsf(data[1])), fmaxf(fabsf(data[2]), fabsf(data[3])));
  const auto abs_max = warp::reduce_max(local_max);
  const auto scale = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
  const auto inv_scale = 1.0f / scale;

  OutStorage result;
  result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
  result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);
  reinterpret_cast<OutStorage*>(value_ptr)[lane_id] = result;
  if (lane_id == 0) *reinterpret_cast<float*>(scale_ptr) = scale;

  PDLTriggerSecondary<kUsePDL>();
}

// =============================================================================
// Host launcher
// =============================================================================
namespace {

template <bool kIsNeox, int32_t kPageBits>
void launch_norm_rope_store(const FusedKIndexerNormRopeStoreParams& params, uint32_t num_blocks, cudaStream_t stream) {
  constexpr auto kernel = fused_k_indexer_norm_rope_store_kernel</*kUsePDL=*/true, kIsNeox, kPageBits>;
  LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kBlockSize, 0, stream, params);
}

template <bool kIsNeox>
void dispatch_norm_rope_store_by_page(const FusedKIndexerNormRopeStoreParams& params, uint32_t num_blocks,
                                      uint32_t page_bits, cudaStream_t stream) {
  switch (page_bits) {
    case 1:
      launch_norm_rope_store<kIsNeox, 1>(params, num_blocks, stream);
      break;
    case 6:
      launch_norm_rope_store<kIsNeox, 6>(params, num_blocks, stream);
      break;
    case 8:
      launch_norm_rope_store<kIsNeox, 8>(params, num_blocks, stream);
      break;
    default:
      KLLM_KERNEL_THROW("fused_k_indexer_norm_rope_store: unsupported page_bits %u (expected 1, 6, or 8)", page_bits);
  }
}

}  // namespace

void fused_k_indexer_norm_rope_store(const void* k_input, void* cache, const void* indices, const void* weight,
                                     const void* bias, const void* freqs_cis, const void* positions,
                                     int64_t k_input_stride_batch, uint32_t batch_size, uint32_t page_bits,
                                     int64_t page_stride, float eps, cudaStream_t stream, bool is_neox) {
  if (batch_size == 0) return;

  const FusedKIndexerNormRopeStoreParams params{
      k_input, cache, indices, weight, bias, freqs_cis, positions, k_input_stride_batch, page_stride, batch_size, eps,
  };
  const uint32_t num_blocks = device::div_ceil(batch_size, kNumWarps);

  if (is_neox) {
    dispatch_norm_rope_store_by_page</*kIsNeox=*/true>(params, num_blocks, page_bits, stream);
  } else {
    dispatch_norm_rope_store_by_page</*kIsNeox=*/false>(params, num_blocks, page_bits, stream);
  }
}

}  // namespace nvidia
}  // namespace llm_kernels
