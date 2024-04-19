/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cstdint>

namespace llm_kernels {
namespace nvidia {
template <typename T>
void InvokeRepetitionPenalty(const T* logits, const T* repetition_penalties, T* output, const int32_t vocab_size,
                             cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels
