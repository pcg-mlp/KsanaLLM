/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

// Expand input[m, n] to output[m, expand_num, n]
template <typename T>
void InvokeExpand(const T* input, T* output, const int32_t m, const int32_t expand_num, const int32_t n,
                  cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
