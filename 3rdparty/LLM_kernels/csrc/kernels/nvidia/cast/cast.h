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

void HalfToFloat(const half* input, int32_t input_length, float* output, cudaStream_t& stream,
                 const size_t input_stride = 1ul, const size_t output_stride = 1ul);

void FloatToHalf(const float* input, int32_t input_length, half* output, cudaStream_t& stream);
void FloatToBFloat16(const float* input, int32_t input_length, __nv_bfloat16* output, cudaStream_t& stream);

void BFloat16ToFloat(const __nv_bfloat16* input, int32_t input_length, float* output, cudaStream_t& stream,
                     const size_t input_stride = 1ul, const size_t output_stride = 1ul);

void FP16ToBFP16(void* data_ptr, int32_t input_length, cudaStream_t& stream);

void BFP16ToFP16(void* data_ptr, int32_t input_length, cudaStream_t& stream);

void ConvertHalfToFloatVectorize(float* output, const half* input, const size_t data_size, cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels
