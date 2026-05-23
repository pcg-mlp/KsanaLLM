/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/moe/moe_fused_gate.cuh
 * Stripped JIT (tvm::ffi) and TensorView wrappers; exposes a plain C++ launcher.
 */
#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/moe_fused_gate.h"

#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/sglang_gate_dtype.cuh"
#include "sgl_kernel/pdl_compat.cuh"
#include "sgl_kernel/utils.cuh"
#include "sgl_kernel/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

namespace llm_kernels::nvidia::sglang_gate {
namespace {

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpsPerCTA = 6;
constexpr uint32_t kSmallTokenThreshold = 512;
constexpr uint32_t kMaxExperts = 512;
constexpr uint32_t kMaxTopK = 16;

enum class ScoringFunc : uint32_t {
  kSigmoid = 0,
  kSqrtSoftplus = 1,
};

template <typename TIn>
struct MoEFusedGateParams {
  const TIn* __restrict__ input;
  const float* __restrict__ bias;
  float* __restrict__ output;
  int32_t* __restrict__ indices;
  uint32_t num_rows;
  uint32_t num_experts;
  uint32_t topk;
  uint32_t num_fused_shared_experts;
  bool renormalize;
  float routed_scaling_factor;
  bool apply_routed_scaling_factor_on_output;
};

template <ScoringFunc kScoringFunc>
__device__ __forceinline__ float compute_score(float x) {
  if constexpr (kScoringFunc == ScoringFunc::kSigmoid) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    return 1.0f / (1.0f + expf(-x));
  } else {
    // sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
    float softplus = log1pf(expf(x));
    return sqrtf(softplus);
  }
}

template <typename TIn, uint32_t kWarpsPerToken, ScoringFunc kScoringFunc>
__global__ void moe_fused_gate_kernel_small_token(const MoEFusedGateParams<TIn> __grid_constant__ params) {
  const auto& [input, bias, output, indices, num_rows, num_experts, topk, num_fused_shared_experts, renormalize, routed_scaling_factor, apply_routed_scaling_factor_on_output] =
      params;

  uint32_t row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  // number of routed experts to select (excluding fused shared experts)
  const uint32_t topk_routed = topk - num_fused_shared_experts;

  uint32_t tid = threadIdx.x;
  uint32_t warp_id = tid / kWarpSize;
  uint32_t lane_id = tid % kWarpSize;

  extern __shared__ float shared_mem[];
  float* shared_scores = shared_mem;
  float* shared_original_scores = shared_mem + num_experts;

  // For warp-level reduction
  __shared__ float warp_maxs[kWarpsPerToken];
  __shared__ int warp_experts[kWarpsPerToken];
  __shared__ int selected_experts[kMaxTopK];

  for (uint32_t e = tid; e < num_experts; e += blockDim.x) {
    float input_val = load_as_float<TIn>(input, row_idx * num_experts + e);
    float bias_val = bias[e];
    float score_val = compute_score<kScoringFunc>(input_val);
    float biased_val = score_val + bias_val;
    shared_scores[e] = biased_val;
    shared_original_scores[e] = score_val;
  }

  __syncthreads();

  // only select topk_routed experts (excluding shared experts)
  for (uint32_t k = 0; k < topk_routed; k++) {
    float my_val = -FLT_MAX;
    int my_expert = -1;
    for (uint32_t e = tid; e < num_experts; e += blockDim.x) {
      if (shared_scores[e] > my_val) {
        my_val = shared_scores[e];
        my_expert = e;
      }
    }

    float warp_max_val = my_val;
    int warp_max_expert = my_expert;

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, warp_max_val, offset);
      int other_expert = __shfl_down_sync(0xFFFFFFFF, warp_max_expert, offset);
      if (other_val > warp_max_val) {
        warp_max_val = other_val;
        warp_max_expert = other_expert;
      }
    }

    if (lane_id == 0 && warp_id < kWarpsPerToken) {
      warp_maxs[warp_id] = warp_max_val;
      warp_experts[warp_id] = warp_max_expert;
    }

    __syncthreads();

    if (warp_id == 0) {
      float final_max = (lane_id < kWarpsPerToken) ? warp_maxs[lane_id] : -FLT_MAX;
      int final_expert = (lane_id < kWarpsPerToken) ? warp_experts[lane_id] : -1;

#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, final_max, offset);
        int other_expert = __shfl_down_sync(0xFFFFFFFF, final_expert, offset);
        if (other_val > final_max) {
          final_max = other_val;
          final_expert = other_expert;
        }
      }

      if (lane_id == 0) {
        selected_experts[k] = final_expert;
      }
    }

    __syncthreads();

    int selected = selected_experts[k];
    if (selected >= 0 && tid == 0) {
      shared_scores[selected] = -FLT_MAX;
    }

    __syncthreads();
  }

  static_assert(kMaxTopK <= device::kWarpThreads);
  if (tid >= device::kWarpThreads) return;

  // only use the first warp to perform write to global operation
  float routed_weight = 0.0f;
  int32_t selected_expert = 0;
  if (tid < topk_routed) {
    int expert_id = selected_experts[tid];
    float score = shared_original_scores[expert_id];
    if (expert_id >= 0 && expert_id < static_cast<int>(num_experts)) {
      routed_weight = score;
      selected_expert = expert_id;
    }
  }
  const auto routed_sum = device::warp::reduce_sum<kMaxTopK>(routed_weight);
  if (tid < topk) {
    const bool is_shared = tid >= topk_routed;
    const auto output_offset = row_idx * topk + tid;
    const auto weight = is_shared ? (routed_sum / routed_scaling_factor) : routed_weight;
    const auto expert_id = is_shared ? (num_experts + tid - topk_routed) : selected_expert;
    const auto scale = apply_routed_scaling_factor_on_output ? routed_scaling_factor : 1.0f;
    const auto norm = renormalize && routed_sum > 0.0f ? routed_sum : 1.0f;
    output[output_offset] = weight / norm * scale;
    indices[output_offset] = expert_id;
  }
}

template <typename TIn, ScoringFunc kScoringFunc>
__global__ void moe_fused_gate_kernel(const MoEFusedGateParams<TIn> __grid_constant__ params) {
  const auto& [input, bias, output, indices, num_rows, num_experts, topk, num_fused_shared_experts, renormalize, routed_scaling_factor, apply_routed_scaling_factor_on_output] =
      params;

  uint32_t row_idx = blockIdx.x * kWarpsPerCTA + threadIdx.y;
  if (row_idx >= num_rows) return;

  // number of routed experts to select (excluding fused shared experts)
  const uint32_t topk_routed = topk - num_fused_shared_experts;

  uint32_t lane_id = threadIdx.x;
  uint32_t warp_id = threadIdx.y;

  extern __shared__ float shared_mem[];
  float* shared_scores = shared_mem + warp_id * num_experts * 2;
  float* shared_original_scores = shared_scores + num_experts;
  __shared__ int selected_experts[kWarpsPerCTA][kMaxTopK];
  int* warp_selected_experts = selected_experts[warp_id];

  for (uint32_t e = lane_id; e < num_experts; e += kWarpSize) {
    float input_val = load_as_float<TIn>(input, row_idx * num_experts + e);
    float bias_val = bias[e];
    float score_val = compute_score<kScoringFunc>(input_val);
    float biased_val = score_val + bias_val;
    shared_scores[e] = biased_val;
    shared_original_scores[e] = score_val;
  }

  __syncwarp();

  // only select topk_routed experts
  for (uint32_t k = 0; k < topk_routed; k++) {
    float max_val = -FLT_MAX;
    int max_expert = -1;

    for (uint32_t expert = lane_id; expert < num_experts; expert += kWarpSize) {
      if (shared_scores[expert] > max_val) {
        max_val = shared_scores[expert];
        max_expert = expert;
      }
    }

    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
      int other_expert = __shfl_down_sync(0xFFFFFFFF, max_expert, offset);

      if (other_val > max_val || (other_val == max_val && other_expert < max_expert)) {
        max_val = other_val;
        max_expert = other_expert;
      }
    }

    if (lane_id == 0) {
      warp_selected_experts[k] = max_expert;
      if (max_expert != -1) {
        shared_scores[max_expert] = -FLT_MAX;
      }
    }

    __syncwarp();
  }

  static_assert(kMaxTopK <= device::kWarpThreads);

  float routed_weight = 0.0f;
  int32_t selected_expert = 0;
  if (lane_id < topk_routed) {
    int expert_id = warp_selected_experts[lane_id];
    if (expert_id >= 0 && expert_id < static_cast<int>(num_experts)) {
      routed_weight = shared_original_scores[expert_id];
      selected_expert = expert_id;
    }
  }
  const auto routed_sum = device::warp::reduce_sum<kMaxTopK>(routed_weight);
  if (lane_id < topk) {
    const bool is_shared = lane_id >= topk_routed;
    const auto output_idx = row_idx * topk + lane_id;
    const auto weight = is_shared ? (routed_sum / routed_scaling_factor) : routed_weight;
    const auto expert_id = is_shared ? (num_experts + lane_id - topk_routed) : selected_expert;
    const auto scale = apply_routed_scaling_factor_on_output ? routed_scaling_factor : 1.0f;
    const auto norm = renormalize && routed_sum > 0.0f ? routed_sum : 1.0f;
    output[output_idx] = weight / norm * scale;
    indices[output_idx] = expert_id;
  }
}

template <typename TIn, uint32_t kNumExperts, uint32_t kTopK, bool kUsePDL>
void launch_score_topk(const MoEFusedGateParams<TIn>& params, cudaStream_t stream) {
  constexpr uint32_t kWarpsPerToken =
      (kNumExperts <= 256) ? 8u : (kNumExperts <= 384 ? 12u : 16u);
  if (params.num_rows <= kSmallTokenThreshold) {
    const uint32_t block_threads = kWarpsPerToken * kWarpSize;
    const size_t smem = 2 * kNumExperts * sizeof(float);
    auto kernel = moe_fused_gate_kernel_small_token<TIn, kWarpsPerToken, ScoringFunc::kSqrtSoftplus>;
    if (kUsePDL) {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr.val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = params.num_rows;
      cfg.blockDim = block_threads;
      cfg.dynamicSmemBytes = smem;
      cfg.stream = stream;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      cudaLaunchKernelEx(&cfg, kernel, params);
    } else {
      kernel<<<params.num_rows, block_threads, smem, stream>>>(params);
    }
  } else {
    const dim3 block_dim(kWarpSize, kWarpsPerCTA);
    const uint32_t grid = (params.num_rows + kWarpsPerCTA - 1) / kWarpsPerCTA;
    const size_t smem = 2 * kNumExperts * sizeof(float) * kWarpsPerCTA;
    auto kernel = moe_fused_gate_kernel<TIn, ScoringFunc::kSqrtSoftplus>;
    if (kUsePDL) {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr.val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = grid;
      cfg.blockDim = block_dim;
      cfg.dynamicSmemBytes = smem;
      cfg.stream = stream;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      cudaLaunchKernelEx(&cfg, kernel, params);
    } else {
      kernel<<<grid, block_dim, smem, stream>>>(params);
    }
  }
}

template <typename TIn>
bool InvokeScoreTopKImpl(const TIn* gating_output, const float* correction_bias, float* topk_weights, int32_t* topk_ids,
                         uint32_t num_tokens, uint32_t num_experts, uint32_t topk, bool renormalize,
                         float routed_scaling_factor, bool use_pdl, cudaStream_t stream) {
  const MoEFusedGateParams<TIn> params{
      .input = gating_output,
      .bias = correction_bias,
      .output = topk_weights,
      .indices = topk_ids,
      .num_rows = num_tokens,
      .num_experts = num_experts,
      .topk = topk,
      .num_fused_shared_experts = 0,
      .renormalize = renormalize,
      .routed_scaling_factor = routed_scaling_factor,
      .apply_routed_scaling_factor_on_output = true,
  };
#define DISPATCH(E, K)                                                          \
  if (num_experts == (E) && topk == (K)) {                                      \
    if (use_pdl) launch_score_topk<TIn, (E), (K), true>(params, stream);        \
    else         launch_score_topk<TIn, (E), (K), false>(params, stream);       \
    return true;                                                                \
  }
  DISPATCH(256, 6)
  DISPATCH(256, 8)
  DISPATCH(384, 6)
  DISPATCH(384, 8)
#undef DISPATCH
  return false;
}

}  // namespace

bool InvokeScoreTopK(const void* gating_output, GateInputDtype input_dtype, const float* correction_bias,
                     float* topk_weights, int32_t* topk_ids, uint32_t num_tokens, uint32_t num_experts, uint32_t topk,
                     bool renormalize, float routed_scaling_factor, bool use_pdl, cudaStream_t stream) {
  if (num_tokens == 0) return true;
  if (input_dtype == GateInputDtype::kFloat32) {
    return InvokeScoreTopKImpl<float>(static_cast<const float*>(gating_output), correction_bias, topk_weights, topk_ids,
                                      num_tokens, num_experts, topk, renormalize, routed_scaling_factor, use_pdl,
                                      stream);
  } else {
    return InvokeScoreTopKImpl<__nv_bfloat16>(static_cast<const __nv_bfloat16*>(gating_output), correction_bias,
                                              topk_weights, topk_ids, num_tokens, num_experts, topk, renormalize,
                                              routed_scaling_factor, use_pdl, stream);
  }
}

}  // namespace llm_kernels::nvidia::sglang_gate
