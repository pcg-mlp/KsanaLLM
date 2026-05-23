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


struct FusedNormRopeStoreParams {
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

constexpr int kNopeDim = 448;  // qk_nope_head_dim = kHeadDim - kRopeDim

// Compressor cache layout constants (per token slot):
//   576 bytes data = 448 FP8 (nope dims, 7 groups x 64) + 128 bf16 (rope dims)
//   8 bytes scale  = 7 x ue8m0 (one per FP8 group) + 1 padding byte
// Page layout: [value_region: 576 * page_size] [scale_region: 8 * page_size]
constexpr int kTokenDataBytes = 576;  // 448 FP8 + 128 bf16
constexpr int kNumQuantBlocks = 7;    // groups 0-6 are FP8, group 7 is bf16
constexpr int kLanesPerGroup = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Main fused kernel
// ─────────────────────────────────────────────────────────────────────────────
// Thread layout (same as fused_norm_rope_kernel):
//   Block: 128 threads = 4 warps x 32 lanes
//   1 warp per work item (token)
//   Each lane holds 2 x 8 = 16 elements of the 512-element head
//
// Lane -> element mapping:
//   vec[0]: lane l owns elements [l*8, l*8+8)         -> covers [0, 256)
//   vec[1]: lane l owns elements [256+l*8, 256+l*8+8) -> covers [256, 512)
//
// FP8 group -> lane mapping (8 lanes per 64-element group):
//   Group 0: lanes 0-7   (vec[0])
//   Group 1: lanes 8-15  (vec[0])
//   Group 2: lanes 16-23 (vec[0])
//   Group 3: lanes 24-31 (vec[0])
//   Group 4: lanes 0-7   (vec[1])
//   Group 5: lanes 8-15  (vec[1])
//   Group 6: lanes 16-23 (vec[1])
//   Group 7: lanes 24-31 (vec[1]) -> bf16, RoPE region
//
template <typename DType, int64_t kHeadDim, int64_t kRopeDim, ForwardMode kMode, uint32_t kPageBits, bool kUsePDL>
__global__ void fused_norm_rope_store_kernel(const __grid_constant__ FusedNormRopeStoreParams params) {
  using namespace device;
  using deepseek_v4::fp8::cast_to_ue8m0;
  using deepseek_v4::fp8::inv_scale_ue8m0;

  static_assert(kHeadDim == 512, "fused_norm_rope_store only supports kHeadDim=512");
  static_assert(kRopeDim == 64, "fused_norm_rope_store only supports kRopeDim=64");
  static_assert(kNopeDim == kHeadDim - kRopeDim, "fused_norm_rope_store only supports kNopeDim=448");
  static_assert(std::is_same_v<DType, __nv_bfloat16>, "fused_norm_rope_store only supports bf16");

  constexpr int64_t kVecSize = 8;    // 16 bytes / sizeof(bf16) = 8
  constexpr int64_t kLocalSize = 2;  // 512 / (32 * 8) = 2

  // RoPE config
  constexpr int64_t kRopeSize = kRopeDim / kVecSize;       // 8 lanes hold rope dims
  constexpr int64_t kNopeSize = kWarpThreads - kRopeSize;  // 24

  const auto& [_input, _weight, eps, num_works, handle, freqs_cis, compress_ratio, cache, indices, page_stride] =
      params;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kFNRNumWarps + warp_id;

  if (work_id >= num_works) return;

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

  // Reload RoPE'd values from smem back into input_vec[1] for rope lanes
  if (lane_id >= kNopeSize) {
    input_vec[kLocalSize - 1] = s_rope_input[warp_id][lane_id - kNopeSize];
  }

  // ── Part 3: FP8 Quantize + Store to paged cache ──────────────────────────
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  void* page_ptr = pointer::offset(cache, page * page_stride);
  void* value_ptr = pointer::offset(page_ptr, offset * kTokenDataBytes);
  void* scale_ptr = pointer::offset(page_ptr, kTokenDataBytes << kPageBits, offset * 8);

  // ── vec[0]: all 32 lanes -> FP8 groups 0-3 ──────────────────────────────
  {
    float local_max = 0.0f;
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      local_max = fmaxf(local_max, fabsf(cast<float>(input_vec[0][j])));
    }
    local_max = fmaxf(local_max, __shfl_xor_sync(warp::kFullMask, local_max, 1));
    local_max = fmaxf(local_max, __shfl_xor_sync(warp::kFullMask, local_max, 2));
    local_max = fmaxf(local_max, __shfl_xor_sync(warp::kFullMask, local_max, 4));

    const float scale_raw = fmaxf(1e-4f, local_max) / math::FP8_E4M3_MAX;
    const int32_t scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const float inv_scale = inv_scale_ue8m0(scale_ue8m0);

    uint8_t out_bytes[kVecSize];
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      float scaled = cast<float>(input_vec[0][j]) * inv_scale;
      out_bytes[j] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
    }
    *reinterpret_cast<uint2*>(pointer::offset(value_ptr, lane_id * kVecSize)) =
        *reinterpret_cast<const uint2*>(out_bytes);

    if (lane_id % kLanesPerGroup == 0) {
      static_cast<uint8_t*>(scale_ptr)[lane_id / kLanesPerGroup] = static_cast<uint8_t>(scale_ue8m0);
    }
  }

  // ── vec[1] lanes 0-23: FP8 groups 4-6 ──────────────────────────────────
  if (lane_id < kNopeSize) {
    float local_max = 0.0f;
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      local_max = fmaxf(local_max, fabsf(cast<float>(input_vec[1][j])));
    }
    local_max = fmaxf(local_max, __shfl_xor_sync(warp::kFullMask, local_max, 1));
    local_max = fmaxf(local_max, __shfl_xor_sync(warp::kFullMask, local_max, 2));
    local_max = fmaxf(local_max, __shfl_xor_sync(warp::kFullMask, local_max, 4));

    const float scale_raw = fmaxf(1e-4f, local_max) / math::FP8_E4M3_MAX;
    const int32_t scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const float inv_scale = inv_scale_ue8m0(scale_ue8m0);

    uint8_t out_bytes[kVecSize];
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      float scaled = cast<float>(input_vec[1][j]) * inv_scale;
      out_bytes[j] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
    }
    *reinterpret_cast<uint2*>(pointer::offset(value_ptr, kWarpThreads * kVecSize + lane_id * kVecSize)) =
        *reinterpret_cast<const uint2*>(out_bytes);

    if (lane_id % kLanesPerGroup == 0) {
      static_cast<uint8_t*>(scale_ptr)[kWarpThreads / kLanesPerGroup + lane_id / kLanesPerGroup] =
          static_cast<uint8_t>(scale_ue8m0);
    }
  } else {
    // ── vec[1] lanes 24-31: bf16 store (group 7, RoPE dims) ────────────────
    const int bf16_byte_offset = kNopeDim + (lane_id - kNopeSize) * static_cast<int>(kVecSize * sizeof(DType));
    store_as<uint4>(pointer::offset(value_ptr, bf16_byte_offset), reinterpret_cast<const uint4&>(input_vec[1]));
    if (lane_id == kNopeSize) {
      static_cast<uint8_t*>(scale_ptr)[kNumQuantBlocks] = 0;
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launcher
// ─────────────────────────────────────────────────────────────────────────────
// Fused RMSNorm + RoPE + FP8 quantize + paged cache store.
//
// `input`       : [num_works, kHeadDim] DType (read-only, NOT modified).
// `weight`      : [kHeadDim] DType, RMSNorm scale.
// `handle`      : depends on mode -
//                   CompressExtend : PrefillPlan* [num_works]
//                   CompressDecode : int32_t*    [num_works] (seq_lens)
//                   DefaultForward : int64_t*    [num_works] (positions)
// `freqs_cis`   : [-1, kRopeDim] DType (split layout: first half cos, second half sin).
// `cache`       : paged FP8 cache base pointer (FlashMLA format).
// `indices`     : [num_works] int32 cache slot indices (-1 = skip store).
// `page_stride` : bytes per page in cache.
template <typename DType, int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageBits>
inline void dispatch_norm_rope_store_mode(const FusedNormRopeStoreParams& params, uint32_t num_blocks, ForwardMode mode,
                                          cudaStream_t stream) {
  switch (mode) {
    case ForwardMode::CompressExtend: {
      constexpr auto kernel =
          fused_norm_rope_store_kernel<DType, kHeadDim, kRopeDim, ForwardMode::CompressExtend, kPageBits,
                                       /*kUsePDL=*/true>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kFNRBlockSize, 0, stream, params);
      break;
    }
    case ForwardMode::CompressDecode: {
      constexpr auto kernel =
          fused_norm_rope_store_kernel<DType, kHeadDim, kRopeDim, ForwardMode::CompressDecode, kPageBits,
                                       /*kUsePDL=*/true>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kFNRBlockSize, 0, stream, params);
      break;
    }
    case ForwardMode::DefaultForward: {
      constexpr auto kernel =
          fused_norm_rope_store_kernel<DType, kHeadDim, kRopeDim, ForwardMode::DefaultForward, kPageBits,
                                       /*kUsePDL=*/true>;
      LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kFNRBlockSize, 0, stream, params);
      break;
    }
  }
}

template <typename DType, int64_t kHeadDim, int64_t kRopeDim = 64>
inline void compress_fused_norm_rope_store(const void* input, const void* weight, const void* handle,
                                           const void* freqs_cis, uint32_t num_works, ForwardMode mode, float eps,
                                           uint32_t compress_ratio, void* cache, const int32_t* indices,
                                           int64_t page_stride, uint32_t page_bits, cudaStream_t stream) {
  if (num_works == 0) return;

  const uint32_t num_blocks = (num_works + kFNRNumWarps - 1u) / kFNRNumWarps;
  const FusedNormRopeStoreParams params{input,     weight,         eps,   num_works, handle,
                                        freqs_cis, compress_ratio, cache, indices,   page_stride};

  switch (page_bits) {
    case 1:
      dispatch_norm_rope_store_mode<DType, kHeadDim, kRopeDim, 1>(params, num_blocks, mode, stream);
      break;
    case 6:
      dispatch_norm_rope_store_mode<DType, kHeadDim, kRopeDim, 6>(params, num_blocks, mode, stream);
      break;
    case 8:
      dispatch_norm_rope_store_mode<DType, kHeadDim, kRopeDim, 8>(params, num_blocks, mode, stream);
      break;
    default:
      KLLM_KERNEL_THROW("compress_fused_norm_rope_store: unsupported page_bits %d (expected 1, 6, or 8)", page_bits);
  }
}

// Explicit instantiations (only kHeadDim=512, bf16)
#define INSTANTIATE_FNR_STORE(DType, kHeadDim)                                                                      \
  template void compress_fused_norm_rope_store<DType, kHeadDim>(const void*, const void*, const void*, const void*, \
                                                                uint32_t, ForwardMode, float, uint32_t, void*,      \
                                                                const int32_t*, int64_t, uint32_t, cudaStream_t);
INSTANTIATE_FNR_STORE(__nv_bfloat16, 512)
#undef INSTANTIATE_FNR_STORE

}  // namespace nvidia
}  // namespace llm_kernels
