/*
 * Adapted from
 * https://github.com/vllm-project/vllm/blob/main/csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu
 *
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/others/vllm/main/deepseek_v4/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.h"

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
constexpr int kRopeDim = 64;      // qk_rope_head_dim
constexpr int kNopeDim = 448;     // qk_nope_head_dim = kHeadDim - kRopeDim
constexpr int kEmbedDim = 32;     // kRopeDim / 2 (number of rotation pairs)
constexpr int kMinNumHeads = 64;  // FlashMLA minimum
constexpr int kWarpSize = 32;
constexpr int kElemsPerLane = kHeadDim / kWarpSize;  // = 16
constexpr int kWarpsPerBlock = 8;
constexpr int kBlockSize = kWarpsPerBlock * kWarpSize;  // = 256

// ─────────────────────────────────────────────────────────────────────────────
// Kernel parameters (passed via __grid_constant__)
// ─────────────────────────────────────────────────────────────────────────────
struct FusedQNormRopeKVInsertParam {
  const __nv_bfloat16* __restrict__ q_in;           // [N, num_heads_q, 512]
  const __nv_bfloat16* __restrict__ kv_in;          // [N, 512]
  void* __restrict__ q_out;                         // [N, 64, 512] (bf16 for kFp8DsMla, fp8 for kFp8E4M3)
  void* __restrict__ swa_kv_cache;                  // paged cache
  const int64_t* __restrict__ positions;            // [N]
  const __nv_bfloat16* __restrict__ cos_sin_cache;  // [max_pos, 64]
  const int32_t* __restrict__ raw_out_loc;          // [N]
  float eps;
  int num_tokens;
  int num_heads_q;
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
// Main fused kernel
// ─────────────────────────────────────────────────────────────────────────────
//
// Grid / block layout:
//   blockDim = 256 (8 warps × 32 lanes)
//   gridDim  = ceil(num_tokens × (num_heads_q + 1) / 8)
//
// Each warp independently handles one (token, slot) pair:
//   slot ∈ [0, num_heads_q)  →  Q branch
//   slot == num_heads_q      →  KV branch
//
// Per-lane data: each lane owns 16 contiguous bf16 elements (512 / 32 = 16).
//
// Q branch (per head), controlled by KV_DTYPE:
//   kFp8E4M3:
//     1. Load 512 bf16 → fp32
//     2. Per-head no-weight RMSNorm
//     3. GPT-J RoPE on dims [448, 512)
//     4. Direct fp32→fp8 cast (no scaling), write to q_out
//   kFp8DsMla (or other):
//     1. Load 512 bf16 → fp32
//     2. Per-head no-weight RMSNorm
//     3. GPT-J RoPE on dims [448, 512)
//     4. Store to q_out with stride = kMinNumHeads(64) × 512
//
// KV branch (single head):
//   1. Load 512 bf16 → fp32
//   2. GPT-J RoPE on dims [448, 512)
//   3. FP8 quantize dims [0, 448): per-64-element UE8M0 scale (4-lane reduction)
//   4. Store to paged SWA cache (format depends on KV_DTYPE)
//
// Cache layout per slot (kFp8DsMla: 576 bytes data + 8 bytes scale):
//   [0,   448)  : FP8-E4M3  (7 groups × 64 elems × 1 byte)
//   [448, 576)  : bf16      (64 elems × 2 bytes, RoPE dims)
//   scale region: 7 UE8M0 bytes + 1 padding, at page_ptr + 576*page_size + offset*8
//
// Cache layout per slot (kFp8E4M3: 512 bytes data, no scales):
//   [0,   448)  : FP8-E4M3  (7 groups × 64 elems × 1 byte, nope dims)
//   [448, 512)  : FP8-E4M3  (1 group  × 64 elems × 1 byte, rope dims)
//
template <uint32_t kPageBits, llm_kernels::utils::KVCacheType KV_DTYPE, bool kUsePDL>
__global__ void fused_qnorm_rope_kv_insert_kernel(const __grid_constant__ FusedQNormRopeKVInsertParam param) {
  using namespace device;
  using deepseek_v4::fp8::cast_to_ue8m0;
  using deepseek_v4::fp8::inv_scale_ue8m0;

  const auto& [q_in, kv_in, q_out, swa_kv_cache, positions, cos_sin_cache, raw_out_loc, eps, num_tokens, num_heads_q,
               page_stride] = param;

  // KV_DTYPE-dependent compile-time constant
  constexpr int kTokenDataBytes = (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) ? 512 : 576;

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  // Dim range this lane owns within the 512-wide head.
  const int dim_base = lane_id * kElemsPerLane;  // in [0, 512) step 16

  // Global warp index → (token_idx, slot)
  // Each token needs (num_heads_q + 1) warps: num_heads_q for Q, 1 for KV
  const int global_warp_id = blockIdx.x * kWarpsPerBlock + warp_id;
  const int total_slots_per_token = num_heads_q + 1;
  const int token_idx = global_warp_id / total_slots_per_token;
  const int slot_idx = global_warp_id % total_slots_per_token;

  if (token_idx >= num_tokens) return;

  float elements[kElemsPerLane];

  PDLWaitPrimary<kUsePDL>();

  const int64_t pos = positions[token_idx];

  if (slot_idx < num_heads_q) {
    // ═══════════════════════════════════════════════════════════════════════
    // Q BRANCH: RMSNorm + RoPE + Write (FP8 dense out-of-place or bf16 expand)
    // ═══════════════════════════════════════════════════════════════════════

    // Load 16 bf16 elements from q_in[token_idx, head_idx, lane_id*16 .. lane_id*16+15]
    const __nv_bfloat16* q_head_ptr =
        q_in + (static_cast<int64_t>(token_idx) * num_heads_q + slot_idx) * kHeadDim + dim_base;
    load_bf16_to_fp32(q_head_ptr, elements);

    // ── Step 1: Per-head no-weight RMSNorm ──────────────────────────────
    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
      sum_sq += elements[i] * elements[i];
    }
    sum_sq = warp::reduce_sum(sum_sq);
    const float rms_rcp = math::rsqrt(sum_sq / static_cast<float>(kHeadDim) + eps);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
      elements[i] *= rms_rcp;
    }

    // ── Step 2: Apply RoPE (GPT-J style) on dims [448, 512) ──────────────
    apply_rope_gptj(elements, dim_base, cos_sin_cache, pos);

    // ── Step 3: Write output ─────────────────────────────────────────────
    if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
      // Dense fp32→fp8 write to q_out (no scaling, downstream kernel reads dense fp8)
      __nv_fp8_e4m3* fp8_out = reinterpret_cast<__nv_fp8_e4m3*>(q_out) +
                               (static_cast<int64_t>(token_idx) * num_heads_q + slot_idx) * kHeadDim + dim_base;
      store_fp32_as_fp8_e4m3(fp8_out, elements, /*inv_scale=*/1.0f);
    } else {
      // Expand Q to [N, kMinNumHeads, kHeadDim] bf16 (FlashMLA)
      __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(q_out) +
                               (static_cast<int64_t>(token_idx) * kMinNumHeads + slot_idx) * kHeadDim + dim_base;
      store_fp32_as_bf16(out_ptr, elements);
    }
  } else {
    // ═══════════════════════════════════════════════════════════════════════
    // KV BRANCH: RoPE + FP8 Quantize + Paged SWA Cache Insert
    // ═══════════════════════════════════════════════════════════════════════

    // Load 16 bf16 elements from kv_in[token_idx, lane_id*16 .. lane_id*16+15]
    const __nv_bfloat16* kv_ptr = kv_in + static_cast<int64_t>(token_idx) * kHeadDim + dim_base;
    load_bf16_to_fp32(kv_ptr, elements);

    // ── Step 1: Apply RoPE (GPT-J style) on dims [448, 512) ──────────────
    apply_rope_gptj(elements, dim_base, cos_sin_cache, pos);

    // ── Step 2: FP8 Quantize + Store to paged cache ─────────────────────
    const int32_t index = raw_out_loc[token_idx];
    if (index == -1) {
      PDLTriggerSecondary<kUsePDL>();
      return;
    }

    const int32_t page = index >> kPageBits;
    const int32_t offset = index & ((1 << kPageBits) - 1);
    const auto page_ptr = pointer::offset(swa_kv_cache, page * page_stride);
    const auto value_ptr = pointer::offset(page_ptr, offset * kTokenDataBytes);

    // Determine which scale group this thread belongs to
    // Thread lane_id handles elements [lane_id*16, lane_id*16+16)
    // Group i handles elements [i*64, (i+1)*64), threads [i*4, i*4+4)

    if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
      // ── kFp8E4M3: direct fp32→fp8 cast, no scales ───────────────────────
      // All 512 bytes stored as FP8: [0,448) nope, [448,512) rope
      __nv_fp8_e4m3* fp8_dst = reinterpret_cast<__nv_fp8_e4m3*>(pointer::offset(value_ptr, dim_base));
      store_fp32_as_fp8_e4m3(fp8_dst, elements, /*inv_scale=*/1.0f);
    } else {
      // ── kFp8DsMla: per-group FP8 (nope) + bf16 (rope) + UE8M0 scales ───
      constexpr uint32_t kFp8LanesMask = 0x0FFFFFFFu;  // lanes 0-27 only, 28-31 take bf16 path
      const int group_id = lane_id >> 2;               // 0..7

      if (group_id < 7) {
        // FP8 path (groups 0-6, nope dims 0-447)
        float local_max = 0.0f;
#pragma unroll
        for (int i = 0; i < kElemsPerLane; ++i) {
          local_max = fmaxf(local_max, fabsf(elements[i]));
        }

        // 4-thread sub-warp reduction for per-group abs_max
        local_max = fmaxf(local_max, __shfl_xor_sync(kFp8LanesMask, local_max, 1));
        local_max = fmaxf(local_max, __shfl_xor_sync(kFp8LanesMask, local_max, 2));

        // Compute UE8M0 scale
        const float scale_raw = fmaxf(1e-4f, local_max) / math::FP8_E4M3_MAX;
        const int32_t scale_ue8m0 = cast_to_ue8m0(scale_raw);
        const float inv_scale = inv_scale_ue8m0(scale_ue8m0);

        // Quantize to FP8 and store
        uint8_t out_bytes[kElemsPerLane];
#pragma unroll
        for (int i = 0; i < kElemsPerLane; ++i) {
          float scaled = elements[i] * inv_scale;
          out_bytes[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
        }
        auto* fp8_dst = reinterpret_cast<uint4*>(pointer::offset(value_ptr, dim_base));
        *fp8_dst = *reinterpret_cast<const uint4*>(out_bytes);

        // Gather scales from all 7 groups to lane 0, write as one 8-byte store
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
          // byte 7 = 0 (padding)
          *static_cast<uint64_t*>(scale_ptr) = packed;
        }
      } else {
        // bf16 path (group 7, rope dims 448-511)
        __nv_bfloat16* bf16_dst = reinterpret_cast<__nv_bfloat16*>(
            pointer::offset(value_ptr, kNopeDim + (dim_base - kNopeDim) * sizeof(__nv_bfloat16)));
        store_fp32_as_bf16(bf16_dst, elements);
      }
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ─────────────────────────────────────────────────────────────────────────────
// Launcher (templated on page_bits and KV_DTYPE)
// ─────────────────────────────────────────────────────────────────────────────
template <uint32_t kPageBits, llm_kernels::utils::KVCacheType KV_DTYPE>
inline void LaunchFusedQNormRopeKVInsert(const FusedQNormRopeKVInsertParam& param, cudaStream_t stream) {
  const int total_warps = param.num_tokens * (param.num_heads_q + /*num_heads_k*/ 1);  // Each warp handles one head dim
  const int num_blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;

  if (num_blocks == 0) return;

  constexpr auto kernel = fused_qnorm_rope_kv_insert_kernel<kPageBits, KV_DTYPE, /*kUsePDL=*/true>;
  LAUNCH_KERNEL_WITH_PDL(kernel, num_blocks, kBlockSize, 0, stream, param);
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────
template <llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeFusedQNormRopeKVInsert(const __nv_bfloat16* q_in, const __nv_bfloat16* kv_in, void* q_out,
                                  void* swa_kv_cache, const int64_t* positions, const __nv_bfloat16* cos_sin_cache,
                                  const int32_t* raw_out_loc, float eps, int num_tokens, int num_heads_q,
                                  int64_t page_stride, int page_bits, cudaStream_t stream) {
  const FusedQNormRopeKVInsertParam param{
      q_in,        kv_in, q_out,      swa_kv_cache, positions,   cos_sin_cache,
      raw_out_loc, eps,   num_tokens, num_heads_q,  page_stride,
  };

  switch (page_bits) {
    case 1:
      LaunchFusedQNormRopeKVInsert<1, KV_DTYPE>(param, stream);
      break;
    case 6:
      LaunchFusedQNormRopeKVInsert<6, KV_DTYPE>(param, stream);
      break;
    case 8:
      LaunchFusedQNormRopeKVInsert<8, KV_DTYPE>(param, stream);
      break;
    default:
      KLLM_KERNEL_THROW("InvokeFusedQNormRopeKVInsert: unsupported page_bits %d (expected 1, 6, or 8)", page_bits);
  }
}

// Explicit instantiations
#define INVOKE_FUSED_DSV4_QNORM_ROPE_KV_INSERT_DECLARATION(KV_DTYPE)                                       \
  template void InvokeFusedQNormRopeKVInsert<KV_DTYPE>(                                                    \
      const __nv_bfloat16* q_in, const __nv_bfloat16* kv_in, void* q_out, void* swa_kv_cache,              \
      const int64_t* positions, const __nv_bfloat16* cos_sin_cache, const int32_t* raw_out_loc, float eps, \
      int num_tokens, int num_heads_q, int64_t page_stride, int page_bits, cudaStream_t stream);
INVOKE_FUSED_DSV4_QNORM_ROPE_KV_INSERT_DECLARATION(llm_kernels::utils::KVCacheType::kFp8DsMla);
INVOKE_FUSED_DSV4_QNORM_ROPE_KV_INSERT_DECLARATION(llm_kernels::utils::KVCacheType::kFp8E4M3);
#undef INVOKE_FUSED_DSV4_QNORM_ROPE_KV_INSERT_DECLARATION

}  // namespace nvidia
}  // namespace llm_kernels
