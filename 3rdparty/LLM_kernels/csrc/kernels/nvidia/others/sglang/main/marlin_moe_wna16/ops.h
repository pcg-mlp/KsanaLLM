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
#include "scalar_type.hpp"

namespace llm_kernels {
namespace nvidia {
namespace marlin_moe_wna16 {

// Scale dtype enum for infer_scale_type
// 0=fp16, 1=bf16, 2=fp8_e4m3fn, 3=fp8_e8m0fnu
::marlin_moe_wna16::ScalarType infer_scale_type(
    const ::marlin_moe_wna16::ScalarType& q_type,
    int scale_dtype,
    int compute_dtype);

void moe_wna16_marlin_gemm(
    void* a,
    void* c,
    void* c_tmp,
    const void* b_q_weight,
    const void* b_bias,
    const void* b_scales,
    const void* global_scale,
    const void* b_zeros,
    const void* g_idx,
    const void* perm,
    void* a_tmp,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_past_padded,
    const float* topk_weights,
    int* workspace,
    int64_t moe_block_size,
    int64_t top_k,
    bool mul_topk_weights,
    bool is_ep,
    ::marlin_moe_wna16::ScalarTypeId b_q_type_id,
    int scale_dtype,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    int64_t num_groups,
    int64_t sorted_token_ids_len,
    int64_t g_idx_len,
    int64_t perm_len,
    int64_t b_zeros_len,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float,
    int compute_dtype,
    cudaStream_t stream);

}  // namespace marlin_moe_wna16
}  // namespace nvidia
}  // namespace llm_kernels
