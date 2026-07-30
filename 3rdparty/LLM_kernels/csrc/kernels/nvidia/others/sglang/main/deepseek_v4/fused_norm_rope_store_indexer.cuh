/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/fused_norm_rope.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/deepseek_v4/fp8_utils.cuh"
#include "csrc/utils/nvidia/assert.h"

namespace llm_kernels {
namespace nvidia {

struct FusedNormRopeStoreIndexerParams {
  const void* __restrict__ input;   // [num_works, kHeadDim] DType (read-only)
  const void* __restrict__ weight;  // [kHeadDim] DType (read-only), RMSNorm scale
  float eps;
  uint32_t num_works;
  const void* __restrict__ handle;     // depends on mode (PrefillPlan* / int32_t* / int64_t*)
  const void* __restrict__ freqs_cis;  // [-1, kRopeDim] DType (split: cos then sin)
  uint32_t compress_ratio;
  // --- Store params ---
  void* __restrict__ cache;             // paged FP8 cache base pointer
  const int32_t* __restrict__ indices;  // [num_works] cache slot indices (-1 = skip)
  int64_t page_stride;                  // bytes per page
};

// Indexer cache layout constants:
//   128 FP8 bytes + 4 bytes float32 scale = 132 bytes total
// Page layout: [value_region: 128 * page_size] [scale_region: 4 * page_size]
constexpr int kIndexerTokenDataBytes = 128;  // 128 FP8 values
constexpr int kIndexerScaleBytes = 4;        // 1 float32 scale per token

// ─────────────────────────────────────────────────────────────────────────────
// Main fused kernel for indexer
// ─────────────────────────────────────────────────────────────────────────────
// Thread layout:
//   Block: 128 threads = 4 warps x 32 lanes
//   1 warp per work item (token)
//   Each lane holds 1 x 4 = 4 elements of the 128-element head
//
// Lane -> element mapping:
//   vec[0]: lane l owns elements [l*4, l*4+4)  -> covers [0, 128)
//
// RoPE region: last 64 dims -> lanes 16-31 (kRopeDim/kVecSize = 64/4 = 16 lanes)
// Nope region: first 64 dims -> lanes 0-15
//
// Quantization: single FP8 group (all 128 dims), full-warp abs-max -> float32 scale.
//
template <typename DType, int64_t kHeadDim, int64_t kRopeDim, ForwardMode kMode, uint32_t kPageBits, bool kUsePDL>
__global__ void fused_norm_rope_store_indexer_kernel(const __grid_constant__ FusedNormRopeStoreIndexerParams params) {
  using namespace device;
  using deepseek_v4::fp8::pack_fp8;

  static_assert(kHeadDim == 128, "fused_norm_rope_store_indexer only supports kHeadDim=128");
  static_assert(kRopeDim == 64, "fused_norm_rope_store_indexer only supports kRopeDim=64");
  static_assert(std::is_same_v<DType, __nv_bfloat16>, "fused_norm_rope_store_indexer only supports bf16");

  constexpr int64_t kVecSize = 4;    // min(8, 128/32) = 4
  constexpr int64_t kLocalSize = 1;  // 128 / (32 * 4) = 1

  // RoPE config
  constexpr int64_t kRopeSize = kRopeDim / kVecSize;       // 16 lanes hold rope dims
  constexpr int64_t kNopeSize = kWarpThreads - kRopeSize;  // 16

  const auto& [_input, _weight, eps, num_works, handle, freqs_cis, compress_ratio, cache, indices, page_stride] =
      params;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kFNRNumWarps + warp_id;

  if (work_id >= num_works) [[unlikely]]
    return;

  // Wait here because reading the plan depends on the previous kernel's write.
  PDLWaitPrimary<kUsePDL>();

  // ── Resolve input pointer, position, and cache index ──
  const DType* input;
  int32_t position;
  int32_t index;
  if constexpr (kMode == ForwardMode::CompressExtend) {
    const auto plan = static_cast<const PrefillPlan*>(handle)[work_id];
    if (plan.ragged_id == 0xFFFFFFFF) [[unlikely]]
      return;
    input = static_cast<const DType*>(_input) + plan.ragged_id * kHeadDim;
    position = plan.position + 1 - compress_ratio;
    index = indices[plan.ragged_id];
  } else if constexpr (kMode == ForwardMode::CompressDecode) {
    input = static_cast<const DType*>(_input) + work_id * kHeadDim;
    const auto seq_len = static_cast<const int32_t*>(handle)[work_id];
    if (seq_len % static_cast<int32_t>(compress_ratio) != 0) return;
    position = seq_len - static_cast<int32_t>(compress_ratio);
    index = indices[work_id];
  } else if constexpr (kMode == ForwardMode::DefaultForward) {
    input = static_cast<const DType*>(_input) + work_id * kHeadDim;
    position = static_cast<const int64_t*>(handle)[work_id];
    index = indices[work_id];
  }

  // Skip norm+rope+quantize entirely if index=-1
  if (index == -1) [[unlikely]] {
    PDLTriggerSecondary<kUsePDL>();
    return;
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

  // ── Part 1: RMSNorm (does NOT write back to input) ──
  const auto gmem = tile::Memory<Storage>::warp();
  Storage input_vec[kLocalSize];
  {
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

    // Save rope lanes' vec to shared memory for Part 2
    if (lane_id >= kNopeSize) {
      s_rope_input[warp_id][lane_id - kNopeSize] = input_vec[kLocalSize - 1];
    }

    __syncwarp();
  }

  // ── Part 2: RoPE (result written back to input_vec for store) ──
  {
    using DTypex2_t = packed_t<DType>;
    const auto mem_elem = tile::Memory<DTypex2_t>::warp();
    const auto elem = mem_elem.load(s_rope_input[warp_id]);
    const auto [x_real, x_imag] = cast<fp32x2_t>(elem);
    const fp32x2_t output = {
        x_real * freq_cos - x_imag * freq_sin,
        x_real * freq_sin + x_imag * freq_cos,
    };
    mem_elem.store(s_rope_input[warp_id], cast<DTypex2_t>(output));

    __syncwarp();
  }

  // Reload RoPE'd values from smem back into input_vec for rope lanes
  if (lane_id >= kNopeSize) {
    input_vec[kLocalSize - 1] = s_rope_input[warp_id][lane_id - kNopeSize];
  }

  // ── Part 3: FP8 Quantize (single group, all 128 dims) + Store ──────────
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  void* page_ptr = pointer::offset(cache, page * page_stride);
  void* value_ptr = pointer::offset(page_ptr, offset * kIndexerTokenDataBytes);
  void* scale_ptr = pointer::offset(page_ptr, kIndexerTokenDataBytes << kPageBits, offset * kIndexerScaleBytes);

  // Cast to fp32x2 pairs first
  using DTypex2_t = packed_t<DType>;
  const auto [x0, x1] = cast<fp32x2_t>(reinterpret_cast<const DTypex2_t&>(input_vec[0][0]));
  const auto [y0, y1] = cast<fp32x2_t>(reinterpret_cast<const DTypex2_t&>(input_vec[0][2]));

  // Full-warp abs-max reduction for single float32 scale
  const float local_max = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(y0), fabsf(y1)));
  const float abs_max = warp::reduce_max(local_max);
  // use normal fp32 scale
  const float scale = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
  const float inv_scale = 1.0f / scale;

  // Quantize and store 4 FP8 bytes per lane = 128 bytes total
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;  // 2 x fp8x2 = 4 bytes
  OutStorage result;
  result[0] = pack_fp8(x0 * inv_scale, x1 * inv_scale);
  result[1] = pack_fp8(y0 * inv_scale, y1 * inv_scale);
  static_cast<OutStorage*>(value_ptr)[lane_id] = result;
  // One lane writes the float32 scale
  if (lane_id == 0) {
    *static_cast<float*>(scale_ptr) = scale;
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launcher
// ─────────────────────────────────────────────────────────────────────────────
// Fused RMSNorm + RoPE + FP8 quantize + paged cache store for indexer.
//
// `input`       : [num_works, kHeadDim] DType (read-only, NOT modified).
// `weight`      : [kHeadDim] DType, RMSNorm scale.
// `handle`      : depends on mode -
//                   CompressExtend : PrefillPlan* [num_works]
//                   CompressDecode : int32_t*    [num_works] (seq_lens)
//                   DefaultForward : int64_t*    [num_works] (positions)
// `freqs_cis`   : [-1, kRopeDim] DType (split layout: first half cos, second half sin).
// `cache`       : paged FP8 cache base pointer (indexer cache format).
// `indices`     : [num_works] int32 cache slot indices (-1 = skip store).
// `page_stride` : bytes per page in cache.
template <typename DType, int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageBits>
inline void dispatch_norm_rope_store_indexer_mode(const FusedNormRopeStoreIndexerParams& params, uint32_t num_blocks,
                                                  ForwardMode mode, cudaStream_t stream) {
  switch (mode) {
    case ForwardMode::CompressExtend: {
      constexpr auto kernel =
          fused_norm_rope_store_indexer_kernel<DType, kHeadDim, kRopeDim, ForwardMode::CompressExtend, kPageBits,
                                               /*kUsePDL=*/true>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kFNRBlockSize, 0, stream, params);
      break;
    }
    case ForwardMode::CompressDecode: {
      constexpr auto kernel =
          fused_norm_rope_store_indexer_kernel<DType, kHeadDim, kRopeDim, ForwardMode::CompressDecode, kPageBits,
                                               /*kUsePDL=*/true>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kFNRBlockSize, 0, stream, params);
      break;
    }
    case ForwardMode::DefaultForward: {
      constexpr auto kernel =
          fused_norm_rope_store_indexer_kernel<DType, kHeadDim, kRopeDim, ForwardMode::DefaultForward, kPageBits,
                                               /*kUsePDL=*/true>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kFNRBlockSize, 0, stream, params);
      break;
    }
  }
}

template <typename DType, int64_t kHeadDim = 128, int64_t kRopeDim = 64>
inline void indexer_fused_norm_rope_store(const void* input, const void* weight, const void* handle,
                                          const void* freqs_cis, uint32_t num_works, ForwardMode mode, float eps,
                                          uint32_t compress_ratio, void* cache, const int32_t* indices,
                                          int64_t page_stride, uint32_t page_bits, cudaStream_t stream) {
  if (num_works == 0) return;

  const uint32_t num_blocks = (num_works + kFNRNumWarps - 1u) / kFNRNumWarps;
  const FusedNormRopeStoreIndexerParams params{input,     weight,         eps,   num_works, handle,
                                               freqs_cis, compress_ratio, cache, indices,   page_stride};

  switch (page_bits) {
    case 1:
      dispatch_norm_rope_store_indexer_mode<DType, kHeadDim, kRopeDim, 1>(params, num_blocks, mode, stream);
      break;
    case 6:
      dispatch_norm_rope_store_indexer_mode<DType, kHeadDim, kRopeDim, 6>(params, num_blocks, mode, stream);
      break;
    case 8:
      dispatch_norm_rope_store_indexer_mode<DType, kHeadDim, kRopeDim, 8>(params, num_blocks, mode, stream);
      break;
    default:
      KLLM_KERNEL_THROW("indexer_fused_norm_rope_store: unsupported page_bits %d (expected 1, 6, or 8)", page_bits);
  }
}

// Explicit instantiation (only kHeadDim=128, bf16)
#define INSTANTIATE_FNR_STORE_INDEXER(DType, kHeadDim)                                                             \
  template void indexer_fused_norm_rope_store<DType, kHeadDim>(const void*, const void*, const void*, const void*, \
                                                               uint32_t, ForwardMode, float, uint32_t, void*,      \
                                                               const int32_t*, int64_t, uint32_t, cudaStream_t);
INSTANTIATE_FNR_STORE_INDEXER(__nv_bfloat16, 128)
#undef INSTANTIATE_FNR_STORE_INDEXER

}  // namespace nvidia
}  // namespace llm_kernels
