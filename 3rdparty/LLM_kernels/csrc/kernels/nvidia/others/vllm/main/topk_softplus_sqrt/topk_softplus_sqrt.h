/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================

This file declares the API for a MoE top-k gate with the sqrt(softplus(x))
activation used by DeepSeek V4. The kernel implementation is ported from vLLM
PR #40760 (`csrc/moe/topk_softplus_sqrt_kernels.cu`), originally adapted by vLLM
from TensorRT-LLM v0.7.1 `moe_kernels.cu`.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <set>

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/thop/utils.h"

namespace llm_kernels {
namespace nvidia {

// num_experts values for which the kernel has a template specialization.
// Mirrors the switch inside topkGatingSoftplusSqrtKernelLauncher; calling
// InvokeTopkSoftplusSqrt with num_experts outside this set throws
// std::runtime_error.
inline const std::set<uint32_t> kTopkSoftplusSqrtSupportedNumExperts = {1,   2,   4,   8,   16,  32,  64, 128,
                                                                        256, 512, 192, 320, 384, 448, 576};

/**
 * @brief DeepSeek V4 MoE top-k gate with sqrt-softplus activation.
 *
 * Supports two paths:
 *   - Hash (input_ids != nullptr && tid2eid != nullptr):
 *       expert_ids[token][k] = tid2eid[input_ids[token]][k]
 *       weights[token][k]    = sqrt(softplus(gating_output[token][expert_id]))
 *                              (optionally renormalized) * routed_scaling_factor
 *   - Score-based (input_ids == nullptr && tid2eid == nullptr):
 *       scores = sqrt(softplus(gating_output));
 *       scores_for_choice = scores (+ correction_bias if non-null)
 *       expert_ids = top-k(scores_for_choice)
 *       weights    = scores.gather(expert_ids)
 *                    (optionally renormalized) * routed_scaling_factor
 *
 * Uses PDL (programmatic stream serialization) on SM90+ for launch overlap.
 *
 * @param input_dtype  dtype of gating_output; must be Float / Float16 / BFloat16.
 *                     Unsupported dtypes throw std::runtime_error.
 * @param index_dtype  dtype of topk_indices / input_ids / tid2eid; must be
 *                     Int (int32) or Long (int64). Unsupported dtypes throw.
 *
 * Throws std::runtime_error on unsupported num_experts / input_dtype / index_dtype.
 */
void InvokeTopkSoftplusSqrt(const void* gating_output, void* topk_weights, void* topk_indices,
                            void* token_expert_indices, int num_tokens, int num_experts, int topk, bool renormalize,
                            double routed_scaling_factor, const float* correction_bias, const void* input_ids,
                            const void* tid2eid, tensorrt_llm::dev::ScalarType input_dtype,
                            tensorrt_llm::dev::ScalarType index_dtype, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
