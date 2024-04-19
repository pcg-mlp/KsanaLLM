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

#include "cast.h"

#include "csrc/utils/nvidia/cuda_bf16_fallbacks.cuh"
#include "csrc/utils/nvidia/cuda_type_utils.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

__global__ void ConvertHalfToFloat(const half* input, float* output, int32_t size) {
  int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    half val = input[idx];
    float floatVal = __half2float(val);
    output[idx] = floatVal;
  }
}

void HalfToFloat(const half* input, int32_t input_length, float* output, cudaStream_t& stream) {
  dim3 grid((input_length + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
  ConvertHalfToFloat<<<grid, block, 0, stream>>>(input, output, input_length);
}

__global__ void ConvertBFloat16ToFloat(const __nv_bfloat16* input, float* output, int32_t size) {
  int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    __nv_bfloat16 val = input[idx];
    float floatVal = __bfloat162float(val);
    output[idx] = floatVal;
  }
}

void BFloat16ToFloat(const __nv_bfloat16* input, int32_t input_length, float* output, cudaStream_t& stream) {
  dim3 grid((input_length + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
  ConvertBFloat16ToFloat<<<grid, block, 0, stream>>>(input, output, input_length);
}

__global__ void ConvertFP16ToBFP16(half* input, __nv_bfloat16* output, int32_t size) {
  int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    float val = __half2float(input[idx]);
    output[idx] = __float2bfloat16(val);
  }
}

void FP16ToBFP16(void* data_ptr, int32_t input_length, cudaStream_t& stream) {
  dim3 grid((input_length + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
  half* input = reinterpret_cast<half*>(data_ptr);
  __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(data_ptr);
  ConvertFP16ToBFP16<<<grid, block, 0, stream>>>(input, output, input_length);
}

__global__ void ConvertBFP16ToFP16(__nv_bfloat16* input, half* output, int32_t size) {
  int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    float val = __bfloat162float(input[idx]);
    output[idx] = __float2half(val);
  }
}

void BFP16ToFP16(void* data_ptr, int32_t input_length, cudaStream_t& stream) {
  dim3 grid((input_length + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
  __nv_bfloat16* input = reinterpret_cast<__nv_bfloat16*>(data_ptr);
  half* output = reinterpret_cast<half*>(data_ptr);
  ConvertBFP16ToFP16<<<grid, block, 0, stream>>>(input, output, input_length);
}

__global__ void ConvertHalf2ToFloat2(const half2* __restrict__ input, float2* __restrict__ output, size_t size) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    output[idx] = CastCudaDataType<float2, half2>(input[idx]);
  }
}

void ConvertHalfToFloatVectorize(float* output, const half* input, const size_t data_size, cudaStream_t& stream) {
  if (data_size % 2 == 0) {
    size_t alignment_data_size = data_size >> 1;
    dim3 grid((alignment_data_size + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM);
    dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
    ConvertHalf2ToFloat2<<<grid, block, 0, stream>>>(reinterpret_cast<const half2*>(input),
                                                     reinterpret_cast<float2*>(output), alignment_data_size);
  } else {
    dim3 grid((data_size + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM);
    dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
    ConvertHalfToFloat<<<grid, block, 0, stream>>>(input, output, data_size);
  }
}

}  // namespace nvidia
}  // namespace llm_kernels
