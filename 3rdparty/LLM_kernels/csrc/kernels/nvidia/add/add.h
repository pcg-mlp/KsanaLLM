/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
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
 */

#pragma once

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

namespace llm_kernels {
namespace nvidia {

template <typename T>
void InvokeAddBiasResidual(T* output, const T* input, const T* residual1, const T* residual2, const T* bias,
                           const float* scale_inter, const float* scale_out, const int32_t m, const int32_t n,
                           cudaStream_t stream);

template <typename T>
void InvokeAddBiasResidual(T* output, const T* residual1, const T* residual2, const T* bias, const int32_t m,
                           const int32_t n, cudaStream_t stream);

template <typename T>
void InvokeAddBiasResidual(T* output, const T* residual1, const T* bias, const int32_t m, const int32_t n,
                           cudaStream_t stream) {
  InvokeAddBiasResidual(output, residual1, (const T*)nullptr, bias, m, n, stream);
}

template <typename T>
void InvokeT5AddResidual(T* output, const T* input, const int32_t m, const int32_t n, cudaStream_t stream);

template <typename T>
void InvokeT5AddBiasResidual(T* output, const T* input, const T* bias, const int32_t m, const int32_t n,
                             cudaStream_t stream);

template <typename T>
void InvokeAddBiasAttentionFfnResidual(T* block_output, const T* ffn_output, const T* attn_output, const T* block_input,
                                       const T* bias, const int32_t m, const int32_t n,
                                       const int32_t block_input_tp_split, cudaStream_t stream);

template <typename T>
void InvokeAddBiasAttentionFfnResidual(T* block_output, const T* ffn_output, const T* attn_output, const T* block_input,
                                       const T* bias, const int32_t m, const int32_t n, cudaStream_t stream) {
  InvokeAddBiasAttentionFfnResidual(block_output, ffn_output, attn_output, block_input, bias, m, n, 1, stream);
}

template <typename T>
void InvokeAddBiasResidualCol32(T* output, const int8_t* input1, const T* input2, const T* bias, int32_t m, int32_t n,
                                cudaStream_t stream, const float* input1_deq_factor_ptr);

template <typename T>
void InvokeAddBiasResidualCol32(T* output, const int32_t* input1, const T* input2, const T* bias, int32_t m, int32_t n,
                                cudaStream_t stream, const float* weight_amax, const float* input1_amax_ptr,
                                const int32_t scale_is_vector = 0);

}  // namespace nvidia
}  // namespace llm_kernels