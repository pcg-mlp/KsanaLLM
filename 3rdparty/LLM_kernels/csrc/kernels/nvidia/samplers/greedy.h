/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cstdint>

namespace llm_kernels {
namespace nvidia {

template <typename T>
void InvokeArgMaxReduce(const T* input, const int32_t batch_size, const int32_t vocab_size, uint32_t* result,
                        cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels
