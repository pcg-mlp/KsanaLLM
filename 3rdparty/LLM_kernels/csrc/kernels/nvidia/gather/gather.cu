/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/gather/gather.h"

#include <cassert>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

using namespace llm_kernels::utils;

template <typename VEC_T>
__device__ __forceinline__ void st_global(const VEC_T* ptr, const VEC_T& value) {
  if constexpr (std::is_same_v<VEC_T, int4>) {
    asm volatile("st.global.v4.s32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
  } else {
    static_assert(std::is_same_v<VEC_T, int2>);
    asm volatile("st.global.v2.s32 [%0], {%1, %2};" : : "l"(ptr), "r"(value.x), "r"(value.y));
  }
}

template <typename VEC_T>
__device__ __forceinline__ VEC_T ld_global_nc(const VEC_T* ptr) {
  VEC_T ret;
  if constexpr (std::is_same_v<VEC_T, int4>) {
    asm volatile("ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(ptr));
  } else {
    static_assert(std::is_same_v<VEC_T, int2>);
    asm volatile("ld.global.nc.v2.s32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
  }
  return ret;
}

template <typename VEC_T>
__global__ void GatherFirstDimKernel(const void* __restrict__ input, const int* __restrict__ indices,
                                     const int64_t data_size, void* __restrict__ output) {
  const int src_index = indices[blockIdx.x];
  const char* input_char = reinterpret_cast<const char*>(input) + src_index * data_size;
  char* output_char = reinterpret_cast<char*>(output) + blockIdx.x * data_size;
  const VEC_T* input_vec = reinterpret_cast<const VEC_T*>(input_char);
  VEC_T* output_vec = reinterpret_cast<VEC_T*>(output_char);
  const int64_t num_vec = data_size / sizeof(VEC_T);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

#pragma unroll
  for (int64_t i = threadIdx.x; i < num_vec; i += blockDim.x) {
    VEC_T input_buf = ld_global_nc<VEC_T>(input_vec + i);
    st_global<VEC_T>(output_vec + i, input_buf);
  }
  const int64_t remain_index = num_vec * sizeof(VEC_T) + threadIdx.x;
  if (remain_index < data_size) {
    output_char[remain_index] = input_char[remain_index];
  }
}

template <>
__global__ void GatherFirstDimKernel<char>(const void* __restrict__ input, const int* __restrict__ indices,
                                           const int64_t data_size, void* __restrict__ output) {
  const int src_index = indices[blockIdx.x];
  const char* input_char = reinterpret_cast<const char*>(input) + src_index * data_size;
  char* output_char = reinterpret_cast<char*>(output) + blockIdx.x * data_size;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

#pragma unroll
  for (int64_t i = threadIdx.x; i < data_size; i += blockDim.x) {
    output_char[i] = input_char[i];
  }
}

void GatherFirstDim(const void* input, const int* indices, const int64_t len, const int64_t data_size, void* output,
                    cudaStream_t stream) {
  dim3 grid(len);
  dim3 block(DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);
  if ((data_size & (sizeof(int4) - 1)) == 0) {
    constexpr auto kernel = GatherFirstDimKernel<int4>;
    LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, input, indices, data_size, output);
  } else if ((data_size & (sizeof(int2) - 1)) == 0) {
    constexpr auto kernel = GatherFirstDimKernel<int2>;
    LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, input, indices, data_size, output);
  } else {
    constexpr auto kernel = GatherFirstDimKernel<char>;
    LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, input, indices, data_size, output);
  }
}

}  // namespace nvidia
}  // namespace llm_kernels
