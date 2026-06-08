/* Copyright 2026 Tencent Inc.  All rights reserved.
 ==============================================================================*/

#include "csrc/kernels/nvidia/qknorm_rope/fused_qknorm_rope.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

static constexpr int kWarpSize = 32;
static constexpr int kWarpsPerBlock = 4;

__global__ void FusedQKNormRoPEKernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int64_t* __restrict__ positions,
    const __nv_bfloat16* __restrict__ cos_sin_cache,
    const int64_t* __restrict__ mask,
    const int num_tokens, const int num_heads, const int head_dim,
    const int stride_size, const float eps) {

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int head_idx = blockIdx.x * kWarpsPerBlock + warp_id;
  const int total_heads = num_tokens * num_heads;
  if (head_idx >= total_heads) return;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  const int token_idx = head_idx / num_heads;

  if (mask != nullptr && mask[token_idx] == 0) return;

  const int head_in_token = head_idx % num_heads;
  const int half_dim = head_dim / 2;
  const int base = token_idx * stride_size + head_in_token * head_dim;
  const int64_t pos = positions[token_idx];
  const int cache_base = pos * head_dim;

  float vals[4];
  float local_sum = 0.0f;
  #pragma unroll
  for (int k = 0; k < 4; k++) {
    int idx = lane_id + k * kWarpSize;
    vals[k] = __bfloat162float(input[base + idx]);
    local_sum += vals[k] * vals[k];
  }

  #pragma unroll
  for (int m = 16; m > 0; m >>= 1) {
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, m, 32);
  }
  const float rsqrt_val = rsqrtf(local_sum / (float)head_dim + eps);

  float normed[4];
  #pragma unroll
  for (int k = 0; k < 4; k++) {
    int idx = lane_id + k * kWarpSize;
    float w = __bfloat162float(norm_weight[idx]);
    normed[k] = vals[k] * rsqrt_val * w;
  }

  float cos0 = __bfloat162float(cos_sin_cache[cache_base + lane_id]);
  float sin0 = __bfloat162float(cos_sin_cache[cache_base + half_dim + lane_id]);
  float out0 = normed[0] * cos0 - normed[2] * sin0;
  float out2 = normed[2] * cos0 + normed[0] * sin0;

  float cos1 = __bfloat162float(cos_sin_cache[cache_base + lane_id + kWarpSize]);
  float sin1 = __bfloat162float(cos_sin_cache[cache_base + half_dim + lane_id + kWarpSize]);
  float out1 = normed[1] * cos1 - normed[3] * sin1;
  float out3 = normed[3] * cos1 + normed[1] * sin1;

  output[base + lane_id] = __float2bfloat16(out0);
  output[base + lane_id + kWarpSize] = __float2bfloat16(out1);
  output[base + lane_id + 64] = __float2bfloat16(out2);
  output[base + lane_id + 96] = __float2bfloat16(out3);

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void InvokeFusedQKNormRoPE(const void* input, void* output, const void* norm_weight,
                           const void* positions, const void* cos_sin_cache,
                           int num_tokens, int num_heads, int head_dim, float eps,
                           cudaStream_t stream, int stride_size,
                           const int64_t* mask) {
  if (num_tokens <= 0 || num_heads <= 0) return;
  if (head_dim != 128) return;
  if (stride_size <= 0) stride_size = num_heads * head_dim;

  const int total_heads = num_tokens * num_heads;
  const int grid_size = (total_heads + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int block_size = kWarpsPerBlock * kWarpSize;

  LAUNCH_KERNEL_WITH_PDL(FusedQKNormRoPEKernel, grid_size, block_size, 0, stream,
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output),
      static_cast<const __nv_bfloat16*>(norm_weight),
      static_cast<const int64_t*>(positions),
      static_cast<const __nv_bfloat16*>(cos_sin_cache),
      mask, num_tokens, num_heads, head_dim, stride_size, eps);
}

}  // namespace nvidia
}  // namespace llm_kernels
