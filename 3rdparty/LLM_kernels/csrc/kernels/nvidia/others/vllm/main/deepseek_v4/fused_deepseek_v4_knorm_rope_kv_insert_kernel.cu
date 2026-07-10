/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/others/vllm/main/deepseek_v4/fused_deepseek_v4_knorm_rope_kv_insert_kernel.h"

#include <cuda_fp8.h>
#include <type_traits>

#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/deepseek_v4/fp8_utils.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/math.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/type.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/utils.cuh"
#include "csrc/kernels/nvidia/others/sglang/main/include/sgl_kernel/warp.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;   // qk_rope_head_dim
constexpr int kNopeDim = 448;  // qk_nope_head_dim = kHeadDim - kRopeDim
constexpr int kEmbedDim = 32;  // kRopeDim / 2 (number of rotation pairs)
constexpr int kWarpSize = 32;
constexpr int kElemsPerLane = kHeadDim / kWarpSize;  // = 16
constexpr int kWarpsPerBlock = 8;
constexpr int kBlockSize = kWarpsPerBlock * kWarpSize;  // = 256

// ─────────────────────────────────────────────────────────────────────────────
// Kernel parameters
// ─────────────────────────────────────────────────────────────────────────────
struct FusedKNormRopeKVInsertParam {
  const __nv_bfloat16* __restrict__ kv_in;           // [N, 512]
  const __nv_bfloat16* __restrict__ kv_norm_weight;  // [512]
  void* __restrict__ swa_kv_cache;                   // paged cache
  const int64_t* __restrict__ positions;             // [N]
  const __nv_bfloat16* __restrict__ cos_sin_cache;   // [max_pos, 64]
  const int32_t* __restrict__ raw_out_loc;           // [N]
  float eps;
  int num_tokens;
  int64_t page_stride;
};

// ─────────────────────────────────────────────────────────────────────────────
// Device helpers
// ─────────────────────────────────────────────────────────────────────────────

// Load 16 bf16 elements from global memory, convert to fp32
__device__ __forceinline__ void load_bf16_to_fp32(const __nv_bfloat16* ptr, float* out) {
  // Load as 4 x float4 (each float4 = 4 x bf16x2 = 8 bf16 values via reinterpret)
  // Use vectorized loads for better memory throughput
  const uint4* vec_ptr = reinterpret_cast<const uint4*>(ptr);
  uint4 data = vec_ptr[0];  // 16 bytes = 8 bf16

  const __nv_bfloat16* bf16_data = reinterpret_cast<const __nv_bfloat16*>(&data);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = __bfloat162float(bf16_data[i]);
  }

  const uint4* vec_ptr2 = reinterpret_cast<const uint4*>(ptr + 8);
  uint4 data2 = vec_ptr2[0];
  const __nv_bfloat16* bf16_data2 = reinterpret_cast<const __nv_bfloat16*>(&data2);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[8 + i] = __bfloat162float(bf16_data2[i]);
  }
}

// Store 16 fp32 elements as bf16 to global memory
__device__ __forceinline__ void store_fp32_as_bf16(__nv_bfloat16* ptr, const float* in) {
  __nv_bfloat16 tmp[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    tmp[i] = __float2bfloat16_rn(in[i]);
  }
  // Store as 2 x uint4 (16 bf16 = 32 bytes)
  uint4* vec_ptr = reinterpret_cast<uint4*>(ptr);
  vec_ptr[0] = reinterpret_cast<const uint4*>(tmp)[0];
  uint4* vec_ptr2 = reinterpret_cast<uint4*>(ptr + 8);
  vec_ptr2[0] = reinterpret_cast<const uint4*>(tmp + 8)[0];
}

// Store 16 fp32 elements as fp8_e4m3 to global memory (vectorized, 16 bytes)
__device__ __forceinline__ void store_fp32_as_fp8_e4m3(__nv_fp8_e4m3* ptr, const float* in, float inv_scale) {
  uint8_t tmp[kElemsPerLane];
#pragma unroll
  for (int i = 0; i < kElemsPerLane; ++i) {
    tmp[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(in[i] * inv_scale, __NV_SATFINITE, __NV_E4M3));
  }
  *reinterpret_cast<uint4*>(ptr) = *reinterpret_cast<const uint4*>(tmp);
}

// GPT-J style RoPE on dims [448, 512). Only lanes 28-31 participate.
// Each lane loads 8 cos + 8 sin via two uint4 vectorized loads, then applies:
//   x_new[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
//   x_new[2i+1] = x[2i] * sin[i] + x[2i+1] * cos[i]
__device__ __forceinline__ void apply_rope_gptj(float* elements, int dim_base, const __nv_bfloat16* cos_sin_cache,
                                                int64_t pos) {
  if (dim_base < kNopeDim) return;

  // cos_sin_cache layout per position: [cos(32), sin(32)]
  const __nv_bfloat16* cos_ptr = cos_sin_cache + pos * kRopeDim;
  const __nv_bfloat16* sin_ptr = cos_ptr + kEmbedDim;
  const int cos_sin_offset = (dim_base - kNopeDim) / 2;

  uint4 cos_data = *reinterpret_cast<const uint4*>(cos_ptr + cos_sin_offset);
  const __nv_bfloat16* cos_bf16 = reinterpret_cast<const __nv_bfloat16*>(&cos_data);

  uint4 sin_data = *reinterpret_cast<const uint4*>(sin_ptr + cos_sin_offset);
  const __nv_bfloat16* sin_bf16 = reinterpret_cast<const __nv_bfloat16*>(&sin_data);

#pragma unroll
  for (int i = 0; i < kElemsPerLane; i += 2) {
    float cos_val = __bfloat162float(cos_bf16[i / 2]);
    float sin_val = __bfloat162float(sin_bf16[i / 2]);
    float x = elements[i];
    float y = elements[i + 1];
    elements[i] = x * cos_val - y * sin_val;
    elements[i + 1] = x * sin_val + y * cos_val;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main fused K-norm + K-rope + K-insert kernel
// ─────────────────────────────────────────────────────────────────────────────
//
// Grid / block layout:
//   blockDim = 256 (8 warps × 32 lanes)
//   gridDim  = ceil(num_tokens / 8)
//
// Each warp independently handles one token.
//
// Per-warp pipeline (only K branch, no Q):
//   1. Load 512 bf16 K elements → fp32
//   2. Per-token weighted RMSNorm (kv_norm_weight applied per element)
//   3. GPT-J RoPE on dims [448, 512)
//   4. FP8 quantize and store to paged SWA cache:
//      kFp8E4M3:  dense fp8 (no scales) over all 512 dims.
//      kFp8DsMla: per-64-element UE8M0 FP8 over [0, 448), bf16 over [448, 512).
//
template <uint32_t kPageBits, llm_kernels::utils::KVCacheType KV_DTYPE, bool kUsePDL>
__global__ void fused_knorm_rope_kv_insert_kernel(const __grid_constant__ FusedKNormRopeKVInsertParam param) {
  using namespace device;
  using deepseek_v4::fp8::cast_to_ue8m0;
  using deepseek_v4::fp8::inv_scale_ue8m0;

  const auto& [kv_in, kv_norm_weight, swa_kv_cache, positions, cos_sin_cache, raw_out_loc, eps, num_tokens,
               page_stride] = param;

  constexpr int kTokenDataBytes = (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) ? 512 : 576;

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int dim_base = lane_id * kElemsPerLane;

  const int token_idx = blockIdx.x * kWarpsPerBlock + warp_id;

  if (token_idx >= num_tokens) return;

  float elements[kElemsPerLane];

  PDLWaitPrimary<kUsePDL>();

  const int64_t pos = positions[token_idx];

  // ── Load 16 bf16 K elements ─────────────────────────────────────────────
  const __nv_bfloat16* kv_ptr = kv_in + static_cast<int64_t>(token_idx) * kHeadDim + dim_base;
  load_bf16_to_fp32(kv_ptr, elements);

  // ── Step 1: Weighted RMSNorm using kv_norm_weight ───────────────────────
  float weight_local[kElemsPerLane];
  load_bf16_to_fp32(kv_norm_weight + dim_base, weight_local);

  float sum_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerLane; ++i) {
    sum_sq += elements[i] * elements[i];
  }
  sum_sq = warp::reduce_sum(sum_sq);
  const float rms_rcp = math::rsqrt(sum_sq / static_cast<float>(kHeadDim) + eps);
#pragma unroll
  for (int i = 0; i < kElemsPerLane; ++i) {
    elements[i] = elements[i] * rms_rcp * weight_local[i];
  }

  // ── Step 2: GPT-J RoPE on dims [448, 512) ───────────────────────────────
  apply_rope_gptj(elements, dim_base, cos_sin_cache, pos);

  // ── Step 3: FP8 quantize + paged store ──────────────────────────────────
  const int32_t index = raw_out_loc[token_idx];
  if (index == -1) {
    PDLTriggerSecondary<kUsePDL>();
    return;
  }

  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  const auto page_ptr = pointer::offset(swa_kv_cache, page * page_stride);
  const auto value_ptr = pointer::offset(page_ptr, offset * kTokenDataBytes);

  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
    // Direct fp32→fp8 cast over the full 512 dims, no scales.
    __nv_fp8_e4m3* fp8_dst = reinterpret_cast<__nv_fp8_e4m3*>(pointer::offset(value_ptr, dim_base));
    store_fp32_as_fp8_e4m3(fp8_dst, elements, /*inv_scale=*/1.0f);
  } else {
    // Per-group FP8 (nope) + bf16 (rope) + UE8M0 scales.
    constexpr uint32_t kFp8LanesMask = 0x0FFFFFFFu;  // lanes 0-27
    const int group_id = lane_id >> 2;               // 0..7

    if (group_id < 7) {
      float local_max = 0.0f;
#pragma unroll
      for (int i = 0; i < kElemsPerLane; ++i) {
        local_max = fmaxf(local_max, fabsf(elements[i]));
      }

      local_max = fmaxf(local_max, __shfl_xor_sync(kFp8LanesMask, local_max, 1));
      local_max = fmaxf(local_max, __shfl_xor_sync(kFp8LanesMask, local_max, 2));

      const float scale_raw = fmaxf(1e-4f, local_max) / math::FP8_E4M3_MAX;
      const int32_t scale_ue8m0 = cast_to_ue8m0(scale_raw);
      const float inv_scale = inv_scale_ue8m0(scale_ue8m0);

      uint8_t out_bytes[kElemsPerLane];
#pragma unroll
      for (int i = 0; i < kElemsPerLane; ++i) {
        float scaled = elements[i] * inv_scale;
        out_bytes[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
      }
      auto* fp8_dst = reinterpret_cast<uint4*>(pointer::offset(value_ptr, dim_base));
      *fp8_dst = *reinterpret_cast<const uint4*>(out_bytes);

      int32_t s0 = __shfl_sync(kFp8LanesMask, scale_ue8m0, 0);
      int32_t s1 = __shfl_sync(kFp8LanesMask, scale_ue8m0, 4);
      int32_t s2 = __shfl_sync(kFp8LanesMask, scale_ue8m0, 8);
      int32_t s3 = __shfl_sync(kFp8LanesMask, scale_ue8m0, 12);
      int32_t s4 = __shfl_sync(kFp8LanesMask, scale_ue8m0, 16);
      int32_t s5 = __shfl_sync(kFp8LanesMask, scale_ue8m0, 20);
      int32_t s6 = __shfl_sync(kFp8LanesMask, scale_ue8m0, 24);
      if (lane_id == 0) {
        void* scale_ptr = pointer::offset(page_ptr, kTokenDataBytes << kPageBits, offset * 8);
        uint64_t packed = static_cast<uint64_t>(static_cast<uint8_t>(s0)) |
                          (static_cast<uint64_t>(static_cast<uint8_t>(s1)) << 8) |
                          (static_cast<uint64_t>(static_cast<uint8_t>(s2)) << 16) |
                          (static_cast<uint64_t>(static_cast<uint8_t>(s3)) << 24) |
                          (static_cast<uint64_t>(static_cast<uint8_t>(s4)) << 32) |
                          (static_cast<uint64_t>(static_cast<uint8_t>(s5)) << 40) |
                          (static_cast<uint64_t>(static_cast<uint8_t>(s6)) << 48);
        *static_cast<uint64_t*>(scale_ptr) = packed;
      }
    } else {
      // bf16 path (group 7, rope dims 448-511)
      __nv_bfloat16* bf16_dst = reinterpret_cast<__nv_bfloat16*>(
          pointer::offset(value_ptr, kNopeDim + (dim_base - kNopeDim) * sizeof(__nv_bfloat16)));
      store_fp32_as_bf16(bf16_dst, elements);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <uint32_t kPageBits, llm_kernels::utils::KVCacheType KV_DTYPE>
inline void LaunchFusedKNormRopeKVInsert(const FusedKNormRopeKVInsertParam& param, cudaStream_t stream) {
  const int total_warps = param.num_tokens;
  const int num_blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;

  if (num_blocks == 0) return;

  constexpr auto kernel = fused_knorm_rope_kv_insert_kernel<kPageBits, KV_DTYPE, /*kUsePDL=*/true>;
  LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kBlockSize, 0, stream, param);
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────
template <llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeFusedKNormRopeKVInsert(const __nv_bfloat16* kv_in, const __nv_bfloat16* kv_norm_weight, void* swa_kv_cache,
                                  const int64_t* positions, const __nv_bfloat16* cos_sin_cache,
                                  const int32_t* raw_out_loc, float eps, int num_tokens, int64_t page_stride,
                                  int page_bits, cudaStream_t stream) {
  const FusedKNormRopeKVInsertParam param{
      kv_in, kv_norm_weight, swa_kv_cache, positions, cos_sin_cache, raw_out_loc, eps, num_tokens, page_stride,
  };

  switch (page_bits) {
    case 1:
      LaunchFusedKNormRopeKVInsert<1, KV_DTYPE>(param, stream);
      break;
    case 6:
      LaunchFusedKNormRopeKVInsert<6, KV_DTYPE>(param, stream);
      break;
    case 8:
      LaunchFusedKNormRopeKVInsert<8, KV_DTYPE>(param, stream);
      break;
    default:
      KLLM_KERNEL_THROW("InvokeFusedKNormRopeKVInsert: unsupported page_bits %d (expected 1, 6, or 8)", page_bits);
  }
}

#define INVOKE_FUSED_DSV4_KNORM_ROPE_KV_INSERT_DECLARATION(KV_DTYPE)                                                  \
  template void InvokeFusedKNormRopeKVInsert<KV_DTYPE>(                                                               \
      const __nv_bfloat16* kv_in, const __nv_bfloat16* kv_norm_weight, void* swa_kv_cache, const int64_t* positions,  \
      const __nv_bfloat16* cos_sin_cache, const int32_t* raw_out_loc, float eps, int num_tokens, int64_t page_stride, \
      int page_bits, cudaStream_t stream);
INVOKE_FUSED_DSV4_KNORM_ROPE_KV_INSERT_DECLARATION(llm_kernels::utils::KVCacheType::kFp8DsMla);
INVOKE_FUSED_DSV4_KNORM_ROPE_KV_INSERT_DECLARATION(llm_kernels::utils::KVCacheType::kFp8E4M3);
#undef INVOKE_FUSED_DSV4_KNORM_ROPE_KV_INSERT_DECLARATION

}  // namespace nvidia
}  // namespace llm_kernels
