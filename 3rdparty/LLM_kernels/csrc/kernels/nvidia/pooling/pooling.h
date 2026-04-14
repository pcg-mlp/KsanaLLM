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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"

namespace llm_kernels {
namespace nvidia {

// L2 normalize each row of the input matrix in-place.
// input: [batch_size, hidden_size], dtype: T (half / __nv_bfloat16 / float)
// The normalization is performed per row: output[i] = input[i] / ||input[i]||_2
template <typename T>
void InvokeL2Normalize(T* data, const int32_t batch_size, const int32_t hidden_size, cudaStream_t stream);

// Mean pooling: average hidden states of each sequence.
// input:   [total_tokens, hidden_size], dtype: T  — the hidden states from all sequences concatenated
// offsets: [batch_size + 1], dtype: size_t         — offsets[i] is the start token index of sequence i;
//                                                    offsets[batch_size] is total_tokens
// output:  [batch_size, hidden_size], dtype: T     — mean-pooled hidden states per sequence
//
// Accumulation uses float32 internally to avoid precision loss with half/bf16.
template <typename T>
void InvokeMeanPooling(const T* input, const size_t* offsets, const int32_t batch_size, const int32_t hidden_size,
                       T* output, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
