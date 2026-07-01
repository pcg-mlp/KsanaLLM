/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 * Copyright 2025 vLLM Team
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
 *
 * Adapted from
 * [vLLM Project] https://github.com/vllm-project/vllm/blob/09e56f92620908d3cf1c3020336460f0db8beead/csrc/moe/moe_ops.h
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Vectorization size used in the sgl_moe_align_block_size kernel
#define SGL_MOE_ALIGN_BLOCK_VEC_SIZE 4

namespace llm_kernels {
namespace nvidia {

template <typename T, typename TOKEN_CNTS_T, bool UseExpertParallel>
void InvokeMoeAlignBlockSize(T* topk_ids, int32_t* sorted_token_ids, int32_t* experts_ids,
                             int32_t* total_tokens_post_pad, const int32_t topk, const int32_t num_experts,
                             const int32_t expert_para_size, const int32_t block_size, const size_t numel,
                             const int32_t rank, const cudaStream_t& stream);

template <typename T>
void InvokeSglMoeAlignBlockSize(T* topk_ids, int32_t* sorted_token_ids, int32_t* expert_ids,
                                int32_t* total_tokens_post_pad, const int32_t max_num_tokens_padded,
                                const int32_t num_experts, const int32_t block_size, const size_t numel,
                                int32_t* cumsum, const cudaStream_t& stream);

// Merged align for two block_m values (saves one topk_ids read + one align kernel launch)
template <typename T>
void InvokeSglMoeAlignBlockSizeMerged(T* topk_ids, int32_t* sorted_token_ids1, int32_t* expert_ids1,
                                      int32_t* total_tokens_post_pad1, int32_t* sorted_token_ids2, int32_t* expert_ids2,
                                      int32_t* total_tokens_post_pad2, const int32_t max_num_tokens_padded1,
                                      const int32_t max_num_tokens_padded2, const int32_t num_experts,
                                      const int32_t block_size1, const int32_t block_size2, const size_t numel,
                                      int32_t* cumsum1, int32_t* cumsum2, const cudaStream_t& stream);

template <typename T>
void InvokeDynamicPerTokenScaledFP8Quant(const T* input, const float* scale, const void* scale_ub, void* output,
                                         const int hidden_size, const int num_tokens, const cudaStream_t& stream);

template <typename T>
void InvokeDynamicScaledFP8Quant(const T* input, float* scale, void* output, const int num_tokens, const int num_elems,
                                 const cudaStream_t& stream);

template <typename T>
void InvokeDynamicScaledFP8Quant(const T* input, float* scale, void* output, const int num_tokens, const int num_elems,
                                 const cudaStream_t& stream);

template <typename T, bool UseExpertParallel>
void InvokeMoeSum(void* input,     // [num_tokens, topk, hidden_size]
                  void* output,    // [num_tokens, hidden_size]
                  void* topk_ids,  // [..., topk]
                  int num_tokens, int topk, int hidden_size, const cudaStream_t& stream);

template <typename T, bool UseExpertParallel, bool HAS_SWIGLU_LIMIT = false>
void SiluAndMul(const T* input, T* output, const int* topk_ids, size_t elements_num, size_t inter_size,
                float swiglu_limit, const cudaStream_t& stream);
// NOTE: flashinfer::activation::act_and_mul_kernel has no swiglu_limit/clamp
// support. Callers that need clamp must route through SiluAndMul (see
// InvokeSiluAndMul's HAS_SWIGLU_LIMIT=true branch).
template <typename T>
void FlashinferSiluAndMul(const T* input, T* output, const int* topk_ids, size_t elements_num, size_t inter_size,
                          const cudaStream_t& stream);
template <typename T, bool UseExpertParallel, bool HAS_SWIGLU_LIMIT = false>
void InvokeSiluAndMul(const T* input, T* output, const int* topk_ids, size_t elements_num, size_t inter_size,
                      float swiglu_limit, const cudaStream_t& stream);

template <typename T>
void InvokeWeightDequant(const uint8_t* x, const float* s, T* output, int M, int N, int block_size,
                         const cudaStream_t& stream);

void InvokeFillIntToBuffer(int* output, void* fill_info, int* fill_info_on_host, int fill_info_length,
                           const cudaStream_t& stream);

// Fused multiply-by-topk-weight + sum across topk experts.
// Replaces InvokeMoeMulTopkWeight + InvokeMoeSum with a single kernel launch.
// input: [num_tokens * topk, hidden_size], topk_weights: [num_tokens * topk] (flat float)
// topk_ids: [num_tokens, topk] (int, for EP mask), output: [num_tokens, hidden_size]
template <typename T, bool UseExpertParallel>
void InvokeMoeFusedMulSum(const void* input, const float* topk_weights, const void* topk_ids, void* output,
                          int num_tokens, int topk, int hidden_size, const cudaStream_t& stream);

// 根据 topk_ids 构建 active expert mask，用于 fused MoE 离线反量化路径跳过未命中专家。
//   topk_ids: int32_t[numel]  - 每个 (token, topk-rank) 槽位上的专家 id
//   active_expert_mask: uint8_t[num_experts] - 输出，命中专家位置写 1，其余写 0
//   numel:    int32_t        - topk_ids 元素个数 (= chunk_num_tokens * topk)
//   num_experts: int32_t     - 本 rank 上的专家数
// 实现：1) cudaMemsetAsync 把 mask 整体清 0
//       2) 启动 scatter kernel，每个 thread 读一个 topk_ids[i]，置 mask[eid] = 1
//          (同 byte 多线程写同值，无竞争且与最终值一致)
template <typename T>
void InvokeBuildActiveExpertMask(const T* topk_ids, uint8_t* active_expert_mask, int32_t numel, int32_t num_experts,
                                 const cudaStream_t& stream);
}  // namespace nvidia
}  // namespace llm_kernels
