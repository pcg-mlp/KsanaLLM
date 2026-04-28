/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {

/**
 * @brief Adapter for DeepEP LowLatency dispatch output to FusedMoeKernel input format
 *
 * This function generates topk_ids (token_selected_slots) and topk_weights
 * (token_final_scales) from DeepEP LowLatency dispatch output for the FusedMoeKernel.
 * It performs the following operations:
 *
 * 1. Creates a mask based on recv_expert_count to identify valid tokens
 * 2. Assigns expert slot IDs to valid tokens (expert_id + ep_rank * num_local_experts)
 * 3. Assigns num_slots to invalid tokens (padding marker)
 * 4. Creates fake top_k=1 format with token_selected_slots [total_tokens, 1]
 * 5. Creates token_final_scales [total_tokens, 1] filled with 1.0
 *
 * @param recv_expert_count Number of valid tokens per expert [num_local_experts]
 * @param token_selected_slots Output expert slot IDs [total_tokens, 1]
 * @param token_final_scales Output routing scales [total_tokens, 1] (all 1.0)
 * @param num_local_experts Number of experts on this rank
 * @param tokens_per_expert Maximum tokens per expert (ep_size * max_tokens_per_rank)
 * @param ep_rank Current expert parallel rank
 * @param num_slots Total number of expert slots across all ranks
 * @param stream CUDA stream for kernel execution
 *
 * @note The output tensors must be pre-allocated with sufficient space:
 *       - token_selected_slots: total_tokens elements
 *       - token_final_scales: total_tokens elements
 *       where total_tokens = num_local_experts * tokens_per_expert
 *
 * @note This function is designed to be called after DeepEP LowLatency dispatch
 *       and before FusedMoeKernel execution.
 */
void InvokeAdaptLowLatencyDispatch(const int32_t* recv_expert_count, int32_t* token_selected_slots,
                                   float* token_final_scales, const int32_t num_local_experts,
                                   const int32_t tokens_per_expert, const int32_t ep_rank, const int32_t num_slots,
                                   const cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels