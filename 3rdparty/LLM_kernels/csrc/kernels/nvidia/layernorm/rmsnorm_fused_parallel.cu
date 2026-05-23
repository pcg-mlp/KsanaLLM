/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/layernorm/rmsnorm_fused_parallel.h"

#include <numeric>  // std::gcd

#include "csrc/kernels/nvidia/common/vec_dtypes.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

__host__ __device__ __forceinline__ constexpr uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__device__ __forceinline__ float shfl_xor_sync(float val, uint32_t mask) {
  return __shfl_xor_sync(0xffffffff, val, mask);
}

// Dispatch vec_size to compile-time constant VEC_SIZE (supported: 1, 2, 4, 8)
#define DISPATCH_VEC_SIZE(vec_size, VEC_SIZE, ...) \
  switch (vec_size) {                              \
    case 8: {                                      \
      constexpr uint32_t VEC_SIZE = 8;             \
      __VA_ARGS__                                  \
      break;                                       \
    }                                              \
    case 4: {                                      \
      constexpr uint32_t VEC_SIZE = 4;             \
      __VA_ARGS__                                  \
      break;                                       \
    }                                              \
    case 2: {                                      \
      constexpr uint32_t VEC_SIZE = 2;             \
      __VA_ARGS__                                  \
      break;                                       \
    }                                              \
    default: {                                     \
      constexpr uint32_t VEC_SIZE = 1;             \
      __VA_ARGS__                                  \
      break;                                       \
    }                                              \
  }

// ============================================================================
// ParallelRmsNormKernel
//   * Design: One block per batch sample. Threads within the block are split
//     proportionally between dim1 and dim2. Each group independently computes
//     RMS norm for its assigned input tensor.
// ============================================================================
template <uint32_t VEC_SIZE, typename T>
__global__ void ParallelRmsNormKernel(const T* __restrict__ input1, const T* __restrict__ input2,
                                      const T* __restrict__ weight1, const T* __restrict__ weight2,
                                      T* __restrict__ output1, T* __restrict__ output2, const uint32_t dim1,
                                      const uint32_t dim2, const uint32_t stride_input1, const uint32_t stride_input2,
                                      const uint32_t stride_output1, const uint32_t stride_output2,
                                      const uint32_t num_threads_1, const uint32_t num_threads_2, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;

  extern __shared__ float smem[];
  float* smem_warp_sum_sq_1 = smem;
  float* smem_warp_sum_sq_2 = smem + num_warps;

  // ========== Phase 1: Compute sum_sq for both inputs in parallel ==========
  float sum_sq = 0.f;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  if (thread_id < num_threads_1) {
    // Process input1
    const uint32_t rounds_1 = ceil_div(dim1, VEC_SIZE * num_threads_1);
    for (uint32_t i = 0; i < rounds_1; i++) {
      vec_t<T, VEC_SIZE> input_vec;
      input_vec.fill(0.f);
      if ((i * num_threads_1 + thread_id) * VEC_SIZE < dim1) {
        input_vec.load(input1 + bx * stride_input1 + i * num_threads_1 * VEC_SIZE + thread_id * VEC_SIZE);
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        sum_sq += float(input_vec[j]) * float(input_vec[j]);
      }
    }
  } else if (thread_id < num_threads_1 + num_threads_2) {
    // Process input2
    const uint32_t local_thread_id = thread_id - num_threads_1;
    const uint32_t rounds_2 = ceil_div(dim2, VEC_SIZE * num_threads_2);
    for (uint32_t i = 0; i < rounds_2; i++) {
      vec_t<T, VEC_SIZE> input_vec;
      input_vec.fill(0.f);
      if ((i * num_threads_2 + local_thread_id) * VEC_SIZE < dim2) {
        input_vec.load(input2 + bx * stride_input2 + i * num_threads_2 * VEC_SIZE + local_thread_id * VEC_SIZE);
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        sum_sq += float(input_vec[j]) * float(input_vec[j]);
      }
    }
  }

  // ========== Phase 2: Warp reduce ==========
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += shfl_xor_sync(sum_sq, offset);
  }

  // Store warp results to appropriate shared memory location
  const uint32_t num_warps_1 = ceil_div(num_threads_1, warp_size);
  const uint32_t num_warps_2 = ceil_div(num_threads_2, warp_size);

  if (tx == 0) {
    if (ty < num_warps_1) {
      smem_warp_sum_sq_1[ty] = sum_sq;
    } else if (ty < num_warps_1 + num_warps_2) {
      smem_warp_sum_sq_2[ty - num_warps_1] = sum_sq;
    }
  }
  __syncthreads();

  // ========== Phase 3: Cross-warp reduce ==========
  if (ty == 0) {
    float sum_sq_1 = 0.f;
    float sum_sq_2 = 0.f;

    // Reduce for input1
    if (tx < num_warps_1) {
      sum_sq_1 = smem_warp_sum_sq_1[tx];
    }
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq_1 += shfl_xor_sync(sum_sq_1, offset);
    }

    // Reduce for input2
    if (tx < num_warps_2) {
      sum_sq_2 = smem_warp_sum_sq_2[tx];
    }
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq_2 += shfl_xor_sync(sum_sq_2, offset);
    }

    if (tx == 0) {
      smem_warp_sum_sq_1[0] = rsqrtf(sum_sq_1 / float(dim1) + eps);
      smem_warp_sum_sq_2[0] = rsqrtf(sum_sq_2 / float(dim2) + eps);
    }
  }
  __syncthreads();

  float rms_rcp_1 = smem_warp_sum_sq_1[0];
  float rms_rcp_2 = smem_warp_sum_sq_2[0];

  // ========== Phase 4: Apply RMSNorm and weights in parallel ==========
  if (thread_id < num_threads_1) {
    // Process output1
    const uint32_t rounds_1 = ceil_div(dim1, VEC_SIZE * num_threads_1);
    for (uint32_t i = 0; i < rounds_1; i++) {
      vec_t<T, VEC_SIZE> input_vec;
      vec_t<T, VEC_SIZE> weight_vec;
      vec_t<T, VEC_SIZE> output_vec;
      input_vec.fill(0.f);
      weight_vec.fill(0.f);
      if ((i * num_threads_1 + thread_id) * VEC_SIZE < dim1) {
        input_vec.load(input1 + bx * stride_input1 + i * num_threads_1 * VEC_SIZE + thread_id * VEC_SIZE);
        weight_vec.load(weight1 + i * num_threads_1 * VEC_SIZE + thread_id * VEC_SIZE);
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        output_vec[j] = float(input_vec[j]) * rms_rcp_1 * float(weight_vec[j]);
      }
      if ((i * num_threads_1 + thread_id) * VEC_SIZE < dim1) {
        output_vec.store(output1 + bx * stride_output1 + i * num_threads_1 * VEC_SIZE + thread_id * VEC_SIZE);
      }
    }
  } else if (thread_id < num_threads_1 + num_threads_2) {
    // Process output2
    const uint32_t local_thread_id = thread_id - num_threads_1;
    const uint32_t rounds_2 = ceil_div(dim2, VEC_SIZE * num_threads_2);
    for (uint32_t i = 0; i < rounds_2; i++) {
      vec_t<T, VEC_SIZE> input_vec;
      vec_t<T, VEC_SIZE> weight_vec;
      vec_t<T, VEC_SIZE> output_vec;
      input_vec.fill(0.f);
      weight_vec.fill(0.f);
      if ((i * num_threads_2 + local_thread_id) * VEC_SIZE < dim2) {
        input_vec.load(input2 + bx * stride_input2 + i * num_threads_2 * VEC_SIZE + local_thread_id * VEC_SIZE);
        weight_vec.load(weight2 + i * num_threads_2 * VEC_SIZE + local_thread_id * VEC_SIZE);
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        output_vec[j] = float(input_vec[j]) * rms_rcp_2 * float(weight_vec[j]);
      }
      if ((i * num_threads_2 + local_thread_id) * VEC_SIZE < dim2) {
        output_vec.store(output2 + bx * stride_output2 + i * num_threads_2 * VEC_SIZE + local_thread_id * VEC_SIZE);
      }
    }
  }
}

// ============================================================================
// Host dispatch + public API
// ============================================================================
template <typename T>
void InvokeParallelRmsNorm(const T* input1, const T* weight1, T* output1, int32_t dim1, int32_t stride_input1,
                           const T* input2, const T* weight2, T* output2, int32_t dim2, int32_t stride_input2,
                           int32_t batch_size, float eps, cudaStream_t stream) {
  const uint32_t udim1 = static_cast<uint32_t>(dim1);
  const uint32_t udim2 = static_cast<uint32_t>(dim2);
  const uint32_t vec_size1 = std::gcd(16u / static_cast<uint32_t>(sizeof(T)), udim1);
  const uint32_t vec_size2 = std::gcd(16u / static_cast<uint32_t>(sizeof(T)), udim2);
  const uint32_t vec_size = std::gcd(vec_size1, vec_size2);

  // Allocate threads proportionally to dim1 and dim2
  const uint32_t min_threads_1 = ceil_div(udim1, vec_size);
  const uint32_t min_threads_2 = ceil_div(udim2, vec_size);

  uint32_t num_threads_1, num_threads_2;
  const uint32_t total_min_threads = min_threads_1 + min_threads_2;

  if (total_min_threads <= 1024) {
    num_threads_1 = min_threads_1;
    num_threads_2 = min_threads_2;
  } else {
    // If > 1024, allocate proportionally
    const float ratio = float(min_threads_1) / float(total_min_threads);
    num_threads_1 = std::max(32u, uint32_t(1024 * ratio));
    num_threads_2 = std::max(32u, 1024 - num_threads_1);
  }

  // Ensure thread counts are warp-aligned
  num_threads_1 = ceil_div(num_threads_1, 32) * 32;
  num_threads_2 = ceil_div(num_threads_2, 32) * 32;

  while (num_threads_1 + num_threads_2 > 1024) {
    if (num_threads_1 > num_threads_2) {
      num_threads_1 -= 32;
    } else {
      num_threads_2 -= 32;
    }
  }

  const uint32_t num_threads = num_threads_1 + num_threads_2;
  const uint32_t num_warps = num_threads / 32;

  dim3 nblks(static_cast<uint32_t>(batch_size));
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = 2 * num_warps * sizeof(float);

  const uint32_t ustride_input1 = static_cast<uint32_t>(stride_input1);
  const uint32_t ustride_input2 = static_cast<uint32_t>(stride_input2);
  // Output is always contiguous (stride == dim)
  const uint32_t ustride_output1 = udim1;
  const uint32_t ustride_output2 = udim2;

  DISPATCH_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = ParallelRmsNormKernel<VEC_SIZE, T>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    LAUNCH_KERNEL_WITH_PDL(kernel, nblks, nthrs, smem_size, stream, input1, input2, weight1, weight2, output1, output2,
                           udim1, udim2, ustride_input1, ustride_input2, ustride_output1, ustride_output2,
                           num_threads_1, num_threads_2, eps);
  });
}

#undef DISPATCH_VEC_SIZE

// Explicit template instantiations
#define INVOKE_PARALLEL_RMS_NORM(T)                                                                         \
  template void InvokeParallelRmsNorm(const T* input1, const T* weight1, T* output1, int32_t dim1,          \
                                      int32_t stride_input1, const T* input2, const T* weight2, T* output2, \
                                      int32_t dim2, int32_t stride_input2, int32_t batch_size, float eps,   \
                                      cudaStream_t stream)
INVOKE_PARALLEL_RMS_NORM(float);
INVOKE_PARALLEL_RMS_NORM(half);
INVOKE_PARALLEL_RMS_NORM(__nv_bfloat16);
#undef INVOKE_PARALLEL_RMS_NORM

}  // namespace nvidia
}  // namespace llm_kernels
