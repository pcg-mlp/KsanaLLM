/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Wrapper for the route_radix kernel, adapted from SGLang PR #32541.
 * The host-side TVM launch wrapper has been replaced by a plain CUDA
 * kernel launch with runtime shape validation.
 */
#include "csrc/kernels/nvidia/others/sglang/main/route_radix/route_radix.h"
#include "csrc/kernels/nvidia/others/sglang/main/route_radix/route_radix.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {
namespace sglang_gate {

template <typename T>
void InvokeSglangRouteRadix(T* scores, const float* bias, float* topk_weights,
                            int32_t* topk_ids, int num_tokens, int num_experts, int topk,
                            bool renormalize, float routed_scaling_factor,
                            bool apply_scale, cudaStream_t stream) {
  if (num_tokens == 0) {
    return;
  }

  // The kernel is hardcoded for K3: 896 experts, top-16.
  KLLM_KERNEL_CHECK_WITH_INFO(
      num_experts == static_cast<int>(kRouteRadixNumExperts) &&
          topk == static_cast<int>(kRouteRadixTopK),
      "route_radix is specialized for N=896, K=16; got N=%d, K=%d",
      num_experts, topk);

  // Vectorized row loads (8B for bf16/fp16, 16B for fp32) need aligned row
  // starts; stride % 4 elements covers both (4 x 2B = 8B / 4 x 4B = 16B).
  KLLM_KERNEL_CHECK_WITH_INFO(
      static_cast<long long>(num_experts) % 4 == 0,
      "route_radix: scores row stride must be a multiple of 4");

  const auto params = RouteRadixParams{
      scores,
      bias,
      topk_weights,
      topk_ids,
      num_tokens,
      static_cast<long long>(num_experts),  // scores_stride
      static_cast<long long>(topk),          // out_w_stride
      static_cast<long long>(topk),          // out_i_stride
      routed_scaling_factor,
      renormalize ? 1 : 0,
      apply_scale ? 1 : 0,
      0,  // sorted = 0 (test sorts by id anyway)
  };

  constexpr bool kUsePDL = true;
  constexpr int kBlockSize = LargeRouterRadixTrait::kBlockSize;

  cudaLaunchConfig_t config;
  config.stream = stream;
  config.gridDim = static_cast<uint32_t>(num_tokens);
  config.blockDim = kBlockSize;
  config.dynamicSmemBytes = 0;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = kUsePDL;
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaError_t launch_err = cudaSuccess;
  if constexpr (std::is_same_v<T, float>) {
    launch_err = cudaLaunchKernelEx(&config, &route_radix_kernel<kUsePDL, fp32_t>, params);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    launch_err = cudaLaunchKernelEx(&config, &route_radix_kernel<kUsePDL, bf16_t>, params);
  } else if constexpr (std::is_same_v<T, half>) {
    launch_err = cudaLaunchKernelEx(&config, &route_radix_kernel<kUsePDL, fp16_t>, params);
  } else {
    KLLM_KERNEL_CHECK_WITH_INFO(false, "route_radix: unsupported score dtype");
  }
  KLLM_KERNEL_CHECK_WITH_INFO(launch_err == cudaSuccess,
                              "route_radix kernel launch failed: %s", cudaGetErrorString(launch_err));
}

// Explicit instantiations for the three supported score dtypes.
template void InvokeSglangRouteRadix<float>(float* scores, const float* bias, float* topk_weights,
                                            int32_t* topk_ids, int num_tokens, int num_experts,
                                            int topk, bool renormalize, float routed_scaling_factor,
                                            bool apply_scale, cudaStream_t stream);

template void InvokeSglangRouteRadix<__nv_bfloat16>(
    __nv_bfloat16* scores, const float* bias, float* topk_weights, int32_t* topk_ids,
    int num_tokens, int num_experts, int topk, bool renormalize, float routed_scaling_factor,
    bool apply_scale, cudaStream_t stream);

template void InvokeSglangRouteRadix<half>(
    half* scores, const float* bias, float* topk_weights, int32_t* topk_ids, int num_tokens,
    int num_experts, int topk, bool renormalize, float routed_scaling_factor, bool apply_scale,
    cudaStream_t stream);

}  // namespace sglang_gate
}  // namespace nvidia
}  // namespace llm_kernels
