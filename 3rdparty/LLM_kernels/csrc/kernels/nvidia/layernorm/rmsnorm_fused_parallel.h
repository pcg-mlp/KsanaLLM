/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {

// Parallel RmsNorm: computes RMSNorm on two inputs with different dims in a single kernel launch.
// input1: [batch_size, dim1] with row stride stride_input1
// input2: [batch_size, dim2] with row stride stride_input2
// output1: [batch_size, dim1] contiguous
// output2: [batch_size, dim2] contiguous
template <typename T>
void InvokeParallelRmsNorm(const T* input1, const T* weight1, T* output1, int32_t dim1, int32_t stride_input1,
                           const T* input2, const T* weight2, T* output2, int32_t dim2, int32_t stride_input2,
                           int32_t batch_size, float eps, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
