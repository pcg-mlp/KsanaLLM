/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/sglang_gate_dtype.cuh"

namespace llm_kernels::nvidia::sglang_gate {

// hash 路径专用 launcher。
//   router_logits: [num_tokens, num_routed_experts], fp32 或 bf16 (由 input_dtype 指定)
//   input_ids:     [num_tokens], int32 (KsanaLLM ModelInput 使用 int32 token IDs)
//   tid2eid:       [vocab, topk], int32
//   topk_weights:  [num_tokens, topk], float
//   topk_ids:      [num_tokens, topk], int32
// 行为对齐 vllm hash 路径：num_shared_experts 固定为 0；indices 写实际 expert id；
// weights = act_sqrt_softplus(logit) / sum_warp(...) * routed_scaling_factor。
void InvokeHashTopK(const void* router_logits, GateInputDtype input_dtype, const int32_t* input_ids,
                    const int32_t* tid2eid, float* topk_weights, int32_t* topk_ids, uint32_t num_tokens,
                    uint32_t num_routed_experts, uint32_t topk, float routed_scaling_factor, bool use_pdl,
                    cudaStream_t stream);

}  // namespace llm_kernels::nvidia::sglang_gate
