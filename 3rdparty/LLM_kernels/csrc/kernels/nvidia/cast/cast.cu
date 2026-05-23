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

#include "csrc/kernels/nvidia/common/vec_dtypes.cuh"
#include "csrc/utils/nvidia/cuda_bf16_fallbacks.cuh"
#include "csrc/utils/nvidia/cuda_type_utils.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

using namespace llm_kernels::utils;

// ============================================================================
// Unified cast kernel using vec_t from vec_dtypes.cuh
// Grid-stride loop with vectorized load/store for maximum bandwidth.
// ============================================================================

// Direct cast: types where vec_t::cast_load/cast_store handles conversion directly
template <typename SrcT, typename DstT, size_t VEC_SIZE>
__global__ void UnifiedCastKernel(const SrcT* __restrict__ input, DstT* __restrict__ output, size_t num_elements) {
  const size_t vec_count = num_elements / VEC_SIZE;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  // Vectorized main loop
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < vec_count; idx += stride) {
    vec_t<DstT, VEC_SIZE> dst;
    dst.cast_load(input + idx * VEC_SIZE);
    dst.store(output + idx * VEC_SIZE);
  }

  // Scalar tail
  const size_t tail_start = vec_count * VEC_SIZE;
  for (size_t idx = tail_start + threadIdx.x + blockIdx.x * blockDim.x; idx < num_elements; idx += stride) {
    vec_t<DstT, 1> dst;
    dst.cast_load(input + idx);
    dst.store(output + idx);
  }
}

// Two-step cast: for types without direct cast_from_impl (e.g., bf16 <-> fp8)
// Goes through float as intermediate.
template <typename SrcT, typename DstT, size_t VEC_SIZE>
__global__ void TwoStepCastKernel(const SrcT* __restrict__ input, DstT* __restrict__ output, size_t num_elements) {
  const size_t vec_count = num_elements / VEC_SIZE;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  // Vectorized main loop: src -> float -> dst
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < vec_count; idx += stride) {
    vec_t<float, VEC_SIZE> tmp;
    tmp.cast_load(input + idx * VEC_SIZE);
    vec_t<DstT, VEC_SIZE> dst;
    dst.cast_from(tmp);
    dst.store(output + idx * VEC_SIZE);
  }

  // Scalar tail
  const size_t tail_start = vec_count * VEC_SIZE;
  for (size_t idx = tail_start + threadIdx.x + blockIdx.x * blockDim.x; idx < num_elements; idx += stride) {
    vec_t<float, 1> tmp;
    tmp.cast_load(input + idx);
    vec_t<DstT, 1> dst;
    dst.cast_from(tmp);
    dst.store(output + idx);
  }
}

// Helper: compute grid/block for grid-stride loop kernels
static inline void GetCastLaunchDims(size_t num_elements, size_t vec_size, dim3& grid, dim3& block) {
  size_t vec_count = num_elements / vec_size;
  size_t work_items = (vec_count > 0) ? vec_count : num_elements;
  size_t grid_size = (work_items + DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM;
  grid_size = std::max(size_t(1), std::min(grid_size, static_cast<size_t>(DEFAULT_CUDA_MAX_BLOCKS_NUM)));
  grid = dim3(static_cast<unsigned>(grid_size));
  block = dim3(DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);
}

// Unified launch helper for direct casts
template <typename SrcT, typename DstT, size_t VEC_SIZE = 8>
static void LaunchUnifiedCast(const SrcT* input, size_t num_elements, DstT* output, cudaStream_t stream) {
  dim3 grid, block;
  GetCastLaunchDims(num_elements, VEC_SIZE, grid, block);
  constexpr auto kernel = UnifiedCastKernel<SrcT, DstT, VEC_SIZE>;
  LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, input, output, num_elements);
}

// Unified launch helper for two-step casts (through float)
template <typename SrcT, typename DstT, size_t VEC_SIZE = 8>
static void LaunchTwoStepCast(const SrcT* input, size_t num_elements, DstT* output, cudaStream_t stream) {
  dim3 grid, block;
  GetCastLaunchDims(num_elements, VEC_SIZE, grid, block);
  constexpr auto kernel = TwoStepCastKernel<SrcT, DstT, VEC_SIZE>;
  LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, input, output, num_elements);
}

// ============================================================================
// Int64 -> Int (vectorized: 2 elements per thread via int4 load -> int2 store)
// int64 is 8 bytes, so 2x int64 = int4 (16 bytes, 128-bit load)
// int is 4 bytes, so 2x int = int2 (8 bytes, 64-bit store)
// ============================================================================

__global__ void ConvertInt64ToIntKernel(const int64_t* __restrict__ input, int* __restrict__ output, size_t size) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  const size_t vec_count = size / 2;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  // Vectorized main loop: 2 elements (128-bit load, 64-bit store) per iteration
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < vec_count; idx += stride) {
    int4 in = reinterpret_cast<const int4*>(input)[idx];
    int2 out;
    out.x = static_cast<int>(reinterpret_cast<const int64_t*>(&in.x)[0]);
    out.y = static_cast<int>(reinterpret_cast<const int64_t*>(&in.z)[0]);
    reinterpret_cast<int2*>(output)[idx] = out;
  }

  // Scalar tail
  const size_t tail_start = vec_count * 2;
  for (size_t idx = tail_start + threadIdx.x + blockIdx.x * blockDim.x; idx < size; idx += stride) {
    output[idx] = static_cast<int>(input[idx]);
  }
}

void Int64ToInt(const int64_t* input, size_t input_length, int* output, cudaStream_t& stream) {
  dim3 grid, block;
  GetCastLaunchDims(input_length, 2, grid, block);
  constexpr auto kernel = ConvertInt64ToIntKernel;
  LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, input, output, input_length);
}

// ============================================================================
// Float -> Half (unified, vec_size=8)
// ============================================================================

void FloatToHalf(const float* input, size_t input_length, half* output, cudaStream_t stream) {
  LaunchUnifiedCast<float, half, 8>(input, input_length, output, stream);
}

// ============================================================================
// Float -> BFloat16 (unified, vec_size=8)
// ============================================================================

void FloatToBFloat16(const float* input, size_t input_length, __nv_bfloat16* output, cudaStream_t stream) {
  LaunchUnifiedCast<float, __nv_bfloat16, 8>(input, input_length, output, stream);
}

// ============================================================================
// Half -> Float (vectorized for stride=1, scalar for strided access)
// ============================================================================

// Strided kernel remains hand-written (vec_t assumes contiguous access)
__global__ void ConvertHalfToFloatStridedKernel(const half* __restrict__ input, float* __restrict__ output,
                                                size_t input_length, size_t input_stride, size_t output_stride) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  if (idx < input_length) {
    size_t input_idx = idx % input_stride;
    if (input_idx < output_stride) {
      output[(idx / input_stride) * output_stride + input_idx] = __half2float(input[idx]);
    }
  }
}

void HalfToFloat(const half* input, size_t input_length, float* output, cudaStream_t stream, const size_t input_stride,
                 const size_t output_stride) {
  if (input_stride == 1 && output_stride == 1) {
    LaunchUnifiedCast<half, float, 8>(input, input_length, output, stream);
  } else {
    dim3 grid(static_cast<unsigned>((input_length + DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM - 1) /
                                    DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM));
    dim3 block(DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);
    constexpr auto kernel = ConvertHalfToFloatStridedKernel;
    LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, input, output, input_length, input_stride, output_stride);
  }
}

// ============================================================================
// BFloat16 -> Float (vectorized for stride=1, scalar for strided access)
// ============================================================================

// Strided kernel remains hand-written
__global__ void ConvertBFloat16ToFloatStridedKernel(const __nv_bfloat16* __restrict__ input, float* __restrict__ output,
                                                    size_t input_length, size_t input_stride, size_t output_stride) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  if (idx < input_length) {
    size_t input_idx = idx % input_stride;
    if (input_idx < output_stride) {
      output[(idx / input_stride) * output_stride + input_idx] = __bfloat162float(input[idx]);
    }
  }
}

void BFloat16ToFloat(const __nv_bfloat16* input, size_t input_length, float* output, cudaStream_t stream,
                     const size_t input_stride, const size_t output_stride) {
  if (input_stride == 1 && output_stride == 1) {
    LaunchUnifiedCast<__nv_bfloat16, float, 8>(input, input_length, output, stream);
  } else {
    dim3 grid(static_cast<unsigned>((input_length + DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM - 1) /
                                    DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM));
    dim3 block(DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);
    constexpr auto kernel = ConvertBFloat16ToFloatStridedKernel;
    LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, input, output, input_length, input_stride, output_stride);
  }
}

// ============================================================================
// Half <-> BFloat16 (in-place, vectorized with grid-stride loop)
// Both types are 2 bytes, so in-place is safe.
// These remain hand-written due to the in-place constraint.
// ============================================================================

__global__ void ConvertHalfToBFloat16Kernel(half* input_output, size_t size) {
  const size_t vec_size = size / 4;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  // Vectorized main loop: 4 elements (64-bit) per iteration
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < vec_size; idx += stride) {
    uint2 packed = reinterpret_cast<uint2*>(input_output)[idx];
    float2 f0 = __half22float2(reinterpret_cast<half2&>(packed.x));
    float2 f1 = __half22float2(reinterpret_cast<half2&>(packed.y));
    __nv_bfloat162 bf0 = __floats2bfloat162_rn(f0.x, f0.y);
    __nv_bfloat162 bf1 = __floats2bfloat162_rn(f1.x, f1.y);
    uint2 out;
    reinterpret_cast<__nv_bfloat162&>(out.x) = bf0;
    reinterpret_cast<__nv_bfloat162&>(out.y) = bf1;
    reinterpret_cast<uint2*>(input_output)[idx] = out;
  }

  // Scalar tail for remaining elements
  for (size_t idx = vec_size * 4 + threadIdx.x + blockIdx.x * blockDim.x; idx < size; idx += stride) {
    half val = input_output[idx];
    reinterpret_cast<__nv_bfloat16*>(input_output)[idx] = __float2bfloat16(__half2float(val));
  }
}

void HalfToBFloat16(void* data_ptr, size_t input_length, cudaStream_t stream) {
  dim3 grid, block;
  GetCastLaunchDims(input_length, 4, grid, block);
  constexpr auto kernel = ConvertHalfToBFloat16Kernel;
  LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, reinterpret_cast<half*>(data_ptr), input_length);
}

__global__ void ConvertBFloat16ToHalfKernel(__nv_bfloat16* input_output, size_t size) {
  const size_t vec_size = size / 4;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  // Vectorized main loop: 4 elements (64-bit) per iteration
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < vec_size; idx += stride) {
    uint2 packed = reinterpret_cast<uint2*>(input_output)[idx];
    float2 f0 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162&>(packed.x));
    float2 f1 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162&>(packed.y));
    half2 h0 = __floats2half2_rn(f0.x, f0.y);
    half2 h1 = __floats2half2_rn(f1.x, f1.y);
    uint2 out;
    reinterpret_cast<half2&>(out.x) = h0;
    reinterpret_cast<half2&>(out.y) = h1;
    reinterpret_cast<uint2*>(input_output)[idx] = out;
  }

  // Scalar tail for remaining elements
  for (size_t idx = vec_size * 4 + threadIdx.x + blockIdx.x * blockDim.x; idx < size; idx += stride) {
    __nv_bfloat16 val = input_output[idx];
    reinterpret_cast<half*>(input_output)[idx] = __float2half(__bfloat162float(val));
  }
}

void BFloat16ToHalf(void* data_ptr, size_t input_length, cudaStream_t stream) {
  dim3 grid, block;
  GetCastLaunchDims(input_length, 4, grid, block);
  constexpr auto kernel = ConvertBFloat16ToHalfKernel;
  LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, reinterpret_cast<__nv_bfloat16*>(data_ptr), input_length);
}

// ============================================================================
// FP8 conversions (unified using vec_t)
// ============================================================================

#if defined(ENABLE_FP8)

// Float <-> FP8 E4M3: direct cast (vec_dtypes.cuh has specialized cast_from_impl)
void FloatToFp8E4M3(const float* input, size_t input_length, __nv_fp8_e4m3* output, cudaStream_t stream) {
  LaunchUnifiedCast<float, __nv_fp8_e4m3, 8>(input, input_length, output, stream);
}

void Fp8E4M3ToFloat(const __nv_fp8_e4m3* input, size_t input_length, float* output, cudaStream_t stream) {
  LaunchUnifiedCast<__nv_fp8_e4m3, float, 8>(input, input_length, output, stream);
}

// Half <-> FP8 E4M3: direct cast (vec_dtypes.cuh has specialized cast_from_impl)
void HalfToFp8E4M3(const half* input, size_t input_length, __nv_fp8_e4m3* output, cudaStream_t stream) {
  LaunchUnifiedCast<half, __nv_fp8_e4m3, 8>(input, input_length, output, stream);
}

void Fp8E4M3ToHalf(const __nv_fp8_e4m3* input, size_t input_length, half* output, cudaStream_t stream) {
  LaunchUnifiedCast<__nv_fp8_e4m3, half, 8>(input, input_length, output, stream);
}

// BFloat16 <-> FP8 E4M3: two-step cast through float (no direct bf16<->fp8 specialization)
void BFloat16ToFp8E4M3(const __nv_bfloat16* input, size_t input_length, __nv_fp8_e4m3* output, cudaStream_t stream) {
  LaunchTwoStepCast<__nv_bfloat16, __nv_fp8_e4m3, 4>(input, input_length, output, stream);
}

void Fp8E4M3ToBFloat16(const __nv_fp8_e4m3* input, size_t input_length, __nv_bfloat16* output, cudaStream_t stream) {
  LaunchTwoStepCast<__nv_fp8_e4m3, __nv_bfloat16, 4>(input, input_length, output, stream);
}

#endif

}  // namespace nvidia
}  // namespace llm_kernels
