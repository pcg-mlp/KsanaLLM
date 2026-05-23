/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <set>
#include <utility>

#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/sglang_gate_dtype.cuh"

namespace llm_kernels::nvidia::sglang_gate {

// 支持的 (num_experts, topk) 组合（窄集）。
inline const std::set<std::pair<uint32_t, uint32_t>> kSglangSupportedExpertTopk = {
    {256, 6}, {256, 8}, {384, 6}, {384, 8}};

/**
 * sglang backend 统一入口。dtype 约束：
 *   gating_output: fp32 OR bf16 logits (由 input_dtype 指定)
 *   topk_indices: int32 only
 *   input_ids (hash path): int32 only (KsanaLLM ModelInput 使用 int32 token IDs)
 *   tid2eid    (hash path): int32 only
 * 不在窄集 / 不满足 dtype 时抛 std::runtime_error。
 * use_hash 由 (input_ids != nullptr && tid2eid != nullptr) 判定。
 */
void InvokeTopkSoftplusSqrtSglang(const void* gating_output, GateInputDtype input_dtype, void* topk_weights,
                                  void* topk_indices, int num_tokens, int num_experts, int topk, bool renormalize,
                                  float routed_scaling_factor, const float* correction_bias, const void* input_ids,
                                  const void* tid2eid, cudaStream_t stream);

}  // namespace llm_kernels::nvidia::sglang_gate
