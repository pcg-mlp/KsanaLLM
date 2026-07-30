/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * This file declares the API for the route_radix MoE gate kernel ported
 * from SGLang PR #32541. The kernel is a specialized K3-only top-k routing
 * kernel hardcoded for 896 experts / top-16. It uses 4-round MSB-first
 * radix select, runs register-resident (224 threads x 4 experts = 896),
 * and is 3.1-3.5x faster than Triton.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {
namespace sglang_gate {

template <typename T>
void InvokeSglangRouteRadix(T* scores, const float* bias, float* topk_weights,
                            int32_t* topk_ids, int num_tokens, int num_experts, int topk,
                            bool renormalize, float routed_scaling_factor,
                            bool apply_scale, cudaStream_t stream);

}  // namespace sglang_gate
}  // namespace nvidia
}  // namespace llm_kernels
