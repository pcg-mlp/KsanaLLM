/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <cuda_runtime.h>
#include <string>

namespace llm_kernels {
namespace nvidia {

/**
 * @brief Performs fused Add and RMSNorm operation
 *
 * @tparam T Input data type (float, half, __nv_bfloat16)
 * @param input Input with shape [m, n] and as output as well because it is in-place operation
 * @param residual resudual with shape [m, n]
 * @param weight norm weight with shape [n]
 * @param eps Epsilon value for numerical stability
 * @param m Number of input rows
 * @param n Number of input columns
 * @param stream CUDA stream
 */
template <typename T>
void InvokeFusedAddRMSNorm(void* input, void* residual, void* weight, double eps, uint32_t m, uint32_t n,
                           cudaStream_t stream);
}  // namespace nvidia
}  // namespace llm_kernels
