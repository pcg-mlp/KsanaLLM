/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/sglang_gate_dtype.cuh"

namespace llm_kernels::nvidia::sglang_gate {

// score 路径专用 launcher（sqrt(softplus(x)) + bias + topk + renormalize）。
//   gating_output:   [num_tokens, num_experts], fp32 或 bf16 (由 input_dtype 指定)
//   correction_bias: [num_experts], float（必须非空；DSv4 noaux_tc 总是带 bias）
//   topk_weights:    [num_tokens, topk], float
//   topk_ids:        [num_tokens, topk], int32
// 行为对齐 vllm score 路径：
//   - bias 仅用于 argmax，weights 用未加 bias 的 sqrt(softplus(x))
//   - renormalize=true 时除以 routed_sum；总是乘 routed_scaling_factor
//   - num_fused_shared_experts=0、apply_routed_scaling_factor_on_output=false 固定
// 支持的窄集：num_experts ∈ {256, 384}, topk ∈ {6, 8}；其他组合返回 false。
[[nodiscard]] bool InvokeScoreTopK(const void* gating_output, GateInputDtype input_dtype,
                                   const float* correction_bias, float* topk_weights, int32_t* topk_ids,
                                   uint32_t num_tokens, uint32_t num_experts, uint32_t topk, bool renormalize,
                                   float routed_scaling_factor, bool use_pdl, cudaStream_t stream);

}  // namespace llm_kernels::nvidia::sglang_gate
