/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================

This file declares the API for a single-group MoE top-k routing kernel ported
from vLLM PR #50000 (`csrc/libtorch_stable/moe/grouped_topk_kernels.cu`).
The kernel supports large expert counts (128-1024 experts) and top-k values
(4-32) via tier-based template dispatch, enabling Kimi K3's 896-expert /
top-16 routing configuration.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {

template <typename T>
bool InvokeVllmSingleGroupTopk(T* scores, float* topk_values, int32_t* topk_indices,
                               const float* bias, int64_t num_tokens, int64_t num_experts,
                               int64_t topk, bool renormalize, double routed_scaling_factor,
                               bool enable_pdl, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
