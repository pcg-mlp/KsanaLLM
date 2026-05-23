/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "expand.h"

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

using namespace llm_kernels::utils;

// Expand kernel: grid = (m, expand_num). Each block copies one input row to one
// output position. Threads stride over the vectorized row.
// This gives maximum parallelism: m * expand_num blocks all independent.
template <typename VEC_T>
__global__ void ExpandKernel(const void* __restrict__ input, void* __restrict__ output, const int n_bytes,
                             const int expand_num) {
  const int row = blockIdx.x;
  const int e = blockIdx.y;

  const VEC_T* input_vec =
      reinterpret_cast<const VEC_T*>(reinterpret_cast<const char*>(input) + static_cast<int64_t>(row) * n_bytes);
  VEC_T* output_vec = reinterpret_cast<VEC_T*>(reinterpret_cast<char*>(output) +
                                                (static_cast<int64_t>(row) * expand_num + e) * n_bytes);

  const int num_vec = n_bytes / sizeof(VEC_T);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

#pragma unroll
  for (int j = threadIdx.x; j < num_vec; j += blockDim.x) {
    output_vec[j] = input_vec[j];
  }
}

template <typename T>
void InvokeExpand(const T* input, T* output, const int32_t m, const int32_t expand_num, const int32_t n,
                  cudaStream_t stream) {
  const int32_t n_bytes = n * sizeof(T);
  dim3 grid(m, expand_num);
  dim3 block(DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);

  if ((n_bytes & (sizeof(int4) - 1)) == 0) {
    constexpr auto kernel = ExpandKernel<int4>;
    LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, input, output, n_bytes, expand_num);
  } else if ((n_bytes & (sizeof(int2) - 1)) == 0) {
    constexpr auto kernel = ExpandKernel<int2>;
    LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, input, output, n_bytes, expand_num);
  } else {
    constexpr auto kernel = ExpandKernel<char>;
    LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, input, output, n_bytes, expand_num);
  }
}

template void InvokeExpand(const float* input, float* output, const int32_t m, const int32_t expand_num,
                           const int32_t n, cudaStream_t stream);
template void InvokeExpand(const half* input, half* output, const int32_t m, const int32_t expand_num, const int32_t n,
                           cudaStream_t stream);
template void InvokeExpand(const __nv_bfloat16* input, __nv_bfloat16* output, const int32_t m, const int32_t expand_num,
                           const int32_t n, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
