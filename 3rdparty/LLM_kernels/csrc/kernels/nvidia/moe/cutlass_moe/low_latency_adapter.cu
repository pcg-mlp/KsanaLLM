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

#include "csrc/utils/nvidia/cuda_utils.h"
#include "low_latency_adapter.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

/**
 * @brief Adapter kernel for DeepEP LowLatency dispatch output
 *
 * This kernel generates topk_ids (token_selected_slots) and topk_weights
 * (token_final_scales) from DeepEP LowLatency dispatch output format
 * for the FusedMoeKernel.
 *
 * Uses shared memory to cache recv_expert_count for reduced global memory
 * accesses. num_local_experts per GPU is always <= 256 (fits in shared memory).
 *
 * Input:
 *   - recv_expert_count: [num_local_experts] - number of valid tokens per expert
 *
 * Output:
 *   - token_selected_slots: [total_tokens, 1] - expert slot IDs
 *   - token_final_scales: [total_tokens, 1] - all ones (fake top_k=1)
 *
 * Algorithm:
 *   1. Load recv_expert_count into shared memory
 *   2. Each thread processes one token position across all experts
 *   3. Check if position < recv_expert_count[expert_id] (valid token)
 *   4. If valid: assign slot_id = expert_id + ep_rank * num_local_experts
 *   5. If invalid: assign slot_id = -1 (padding marker)
 *
 * @param recv_expert_count Number of valid tokens per expert [num_local_experts]
 * @param token_selected_slots Output slot IDs [total_tokens, 1]
 * @param token_final_scales Output scales [total_tokens, 1] (all 1.0)
 * @param num_local_experts Number of experts on this rank
 * @param tokens_per_expert Maximum tokens per expert (ep_size * max_tokens)
 * @param ep_rank Current expert parallel rank
 * @param num_slots Total number of expert slots across all ranks
 */
__global__ void adapt_low_latency_dispatch_kernel(const int32_t* __restrict__ recv_expert_count,
                                                  int32_t* __restrict__ token_selected_slots,
                                                  float* __restrict__ token_final_scales,
                                                  const int32_t num_local_experts, const int32_t tokens_per_expert,
                                                  const int32_t ep_rank, const int32_t num_slots) {
  // Shared memory for recv_expert_count
  extern __shared__ int32_t shared_recv_count[];

  // Load recv_expert_count into shared memory
  if (threadIdx.x < num_local_experts) {
    shared_recv_count[threadIdx.x] = recv_expert_count[threadIdx.x];
  }
  __syncthreads();

  const int32_t total_positions = num_local_experts * tokens_per_expert;
  const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= total_positions) return;

  // Decode 2D position: which expert and which token within that expert
  const int32_t expert_id = tid / tokens_per_expert;
  const int32_t token_pos = tid % tokens_per_expert;

  // Use shared memory for recv_expert_count lookup
  const bool is_valid = token_pos < shared_recv_count[expert_id];

  // Assign slot ID: valid tokens get local expert slot, invalid get -1
  const int32_t slot_id = is_valid ? (expert_id + ep_rank * num_local_experts) : -1;

  // Write slot ID (fake top_k=1 format)
  token_selected_slots[tid] = slot_id;

  // Write scale (all 1.0 for fake top_k=1)
  token_final_scales[tid] = 1.0f;
}

// Host function to launch the kernel
void InvokeAdaptLowLatencyDispatch(const int32_t* recv_expert_count, int32_t* token_selected_slots,
                                   float* token_final_scales, const int32_t num_local_experts,
                                   const int32_t tokens_per_expert, const int32_t ep_rank, const int32_t num_slots,
                                   const cudaStream_t& stream) {
  const int32_t total_positions = num_local_experts * tokens_per_expert;

  constexpr int BLOCK_SIZE = 256;
  const int num_blocks = (total_positions + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int shared_mem_size = num_local_experts * sizeof(int32_t);

  adapt_low_latency_dispatch_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size, stream>>>(
      recv_expert_count, token_selected_slots, token_final_scales, num_local_experts, tokens_per_expert, ep_rank,
      num_slots);

  CHECK_NVIDIA_CUDA_ERROR(cudaGetLastError());
}

}  // namespace nvidia
}  // namespace llm_kernels
