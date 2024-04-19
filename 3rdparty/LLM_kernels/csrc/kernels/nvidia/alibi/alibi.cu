/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

__global__ void GetAlibiSlopesKernel(float* slopes, int total_num_heads) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < total_num_heads) {
    int closest_power_of_2 = pow(2, floor(log2f(total_num_heads)));
    float base = pow(2, -(pow(2, -(log2f(closest_power_of_2) - 3))));
    if (idx < closest_power_of_2) {
      slopes[idx] = pow(base, idx + 1);
    } else if (closest_power_of_2 != total_num_heads) {
      float extra_base = pow(2, -(pow(2, -(log2f(2 * closest_power_of_2) - 3))));
      int num_remaining_heads = min(closest_power_of_2, total_num_heads - closest_power_of_2);
      if (idx - closest_power_of_2 < num_remaining_heads) {
        slopes[idx] = pow(extra_base, 2 * (idx - closest_power_of_2) + 1);
      }
    }
  }
}

void GetAlibiSlopesCuda(float* slopes, int total_num_heads, cudaStream_t& stream) {
  dim3 block(std::min(total_num_heads, 512), 1, 1);
  dim3 grid((total_num_heads + block.x - 1) / block.x, 1, 1);
  GetAlibiSlopesKernel<<<grid, block, 0, stream>>>(slopes, total_num_heads);
}

}  // namespace nvidia
}  // namespace llm_kernels
