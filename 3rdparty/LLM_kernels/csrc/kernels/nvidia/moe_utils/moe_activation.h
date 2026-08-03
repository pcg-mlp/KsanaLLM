/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

template <typename T, bool UseExpertParallel>
void InvokeSituAndMul(const T* input, T* output, const int* topk_ids, size_t elements_num, size_t inter_size,
                      float beta, float linear_beta, const cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels
