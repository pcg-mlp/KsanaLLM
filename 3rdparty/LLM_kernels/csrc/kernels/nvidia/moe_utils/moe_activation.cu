/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#include "moe_activation.h"

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

template <typename T, bool UseExpertParallel>
__global__ void situ_and_mul_kernel(const T* x, T* out, const int* topk_ids, size_t size, size_t inter_size, float beta,
                                    float linear_beta) {
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= size) return;
  if constexpr (UseExpertParallel) {
    if (topk_ids[token_idx] < 0) return;
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = (linear_beta > 0.0f) ? 1.0f / linear_beta : 0.0f;
  for (int64_t idx = threadIdx.x; idx < inter_size; idx += blockDim.x) {
    float a = static_cast<float>(__ldg(&x[token_idx * 2 * inter_size + idx]));
    float b = static_cast<float>(__ldg(&x[token_idx * 2 * inter_size + inter_size + idx]));
    float gate_out = beta * tanhf(a * inv_beta) / (1.0f + expf(-a));
    float up_out = (linear_beta > 0.0f) ? linear_beta * tanhf(b * inv_linear_beta) : b;
    out[token_idx * inter_size + idx] = static_cast<T>(gate_out * up_out);
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, bool UseExpertParallel>
void InvokeSituAndMul(const T* input, T* output, const int* topk_ids, size_t elements_num, size_t inter_size,
                      float beta, float linear_beta, const cudaStream_t& stream) {
  size_t num_tokens = elements_num / 2 / inter_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(inter_size, static_cast<size_t>(1024)));
  auto* kernel = situ_and_mul_kernel<T, UseExpertParallel>;
  LAUNCH_KERNEL_WITH_PDL(kernel, grid, block, 0, stream, input, output, topk_ids, num_tokens, inter_size, beta,
                         linear_beta);
}

#define INVOKE_SITU_AND_MUL(T)                                                                    \
  template void InvokeSituAndMul<T, true>(const T*, T*, const int*, size_t, size_t, float, float, \
                                          const cudaStream_t&);                                   \
  template void InvokeSituAndMul<T, false>(const T*, T*, const int*, size_t, size_t, float, float, const cudaStream_t&)
INVOKE_SITU_AND_MUL(float);
INVOKE_SITU_AND_MUL(half);
INVOKE_SITU_AND_MUL(__nv_bfloat16);
#undef INVOKE_SITU_AND_MUL

}  // namespace nvidia
}  // namespace llm_kernels
