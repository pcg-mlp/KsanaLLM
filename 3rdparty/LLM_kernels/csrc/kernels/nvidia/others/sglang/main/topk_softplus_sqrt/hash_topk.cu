/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/hash_topk.cuh
 * Stripped JIT (tvm::ffi) and TensorView wrappers; exposes a plain C++ launcher.
 */
#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/hash_topk.h"

#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/sglang_gate_dtype.cuh"
#include "sgl_kernel/pdl_compat.cuh"
#include "sgl_kernel/utils.cuh"
#include "sgl_kernel/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels::nvidia::sglang_gate {
namespace {

SGL_DEVICE float act_sqrt_softplus(float x) {
  const float softplus = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
  return sqrtf(softplus);
}

template <typename TIn>
struct MoEHashTopKParams {
  const TIn* __restrict__ router_logits;
  const int32_t* __restrict__ input_id;
  const int32_t* __restrict__ tid2eid;
  int32_t* __restrict__ topk_ids;
  float* __restrict__ topk_weights;
  uint32_t num_tokens;
  uint32_t topk;
  uint32_t num_routed_experts;
  float routed_scaling_factor;
};

template <typename TIn, bool kUsePDL>
__global__ void moe_hash_topk_fused(const MoEHashTopKParams<TIn> __grid_constant__ params) {
  const auto& [router_logits, input_id, tid2eid, topk_ids, topk_weights, num_tokens, topk, num_routed_experts,
               routed_scaling_factor] = params;

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = tid / device::kWarpThreads;
  const uint32_t lane_id = tid % device::kWarpThreads;
  if (warp_id >= num_tokens) return;
  const auto token_id = input_id[warp_id];

  PDLWaitPrimary<kUsePDL>();

  float routed_weight = 0.0f;
  int32_t expert_id = 0;
  if (lane_id < topk) {
    expert_id = tid2eid[token_id * topk + lane_id];
    routed_weight =
        act_sqrt_softplus(load_as_float<TIn>(router_logits, warp_id * num_routed_experts + expert_id));
  }

  const auto routed_sum = device::warp::reduce_sum(routed_weight);
  if (lane_id < topk) {
    const auto output_offset = warp_id * topk + lane_id;
    topk_ids[output_offset] = expert_id;
    topk_weights[output_offset] = (routed_weight / routed_sum) * routed_scaling_factor;
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename TIn>
void launch_hash_topk(const TIn* router_logits, const int32_t* input_ids, const int32_t* tid2eid, float* topk_weights,
                     int32_t* topk_ids, uint32_t num_tokens, uint32_t num_routed_experts, uint32_t topk,
                     float routed_scaling_factor, bool use_pdl, cudaStream_t stream) {
  const MoEHashTopKParams<TIn> params{
      .router_logits = router_logits,
      .input_id = input_ids,
      .tid2eid = tid2eid,
      .topk_ids = topk_ids,
      .topk_weights = topk_weights,
      .num_tokens = num_tokens,
      .topk = topk,
      .num_routed_experts = num_routed_experts,
      .routed_scaling_factor = routed_scaling_factor,
  };
  constexpr uint32_t kBlockSize = 128u;
  constexpr uint32_t kNumWarps = kBlockSize / 32u;
  const uint32_t num_blocks = (num_tokens + kNumWarps - 1) / kNumWarps;
  if (use_pdl) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = num_blocks;
    cfg.blockDim = kBlockSize;
    cfg.stream = stream;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, moe_hash_topk_fused<TIn, true>, params);
  } else {
    moe_hash_topk_fused<TIn, false><<<num_blocks, kBlockSize, 0, stream>>>(params);
  }
}

}  // namespace

void InvokeHashTopK(const void* router_logits, GateInputDtype input_dtype, const int32_t* input_ids,
                    const int32_t* tid2eid, float* topk_weights, int32_t* topk_ids, uint32_t num_tokens,
                    uint32_t num_routed_experts, uint32_t topk, float routed_scaling_factor, bool use_pdl,
                    cudaStream_t stream) {
  if (num_tokens == 0) return;
  if (input_dtype == GateInputDtype::kFloat32) {
    launch_hash_topk<float>(static_cast<const float*>(router_logits), input_ids, tid2eid, topk_weights, topk_ids,
                            num_tokens, num_routed_experts, topk, routed_scaling_factor, use_pdl, stream);
  } else {
    launch_hash_topk<__nv_bfloat16>(static_cast<const __nv_bfloat16*>(router_logits), input_ids, tid2eid, topk_weights,
                                    topk_ids, num_tokens, num_routed_experts, topk, routed_scaling_factor, use_pdl,
                                    stream);
  }
}

}  // namespace llm_kernels::nvidia::sglang_gate
