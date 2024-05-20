/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

template <typename T>
void AssembleLastToken(const T* input, const size_t* ids_offsets, const size_t* prefix_offsets,
                       const int32_t batch_size, const int32_t hidden_units_num, T* output, cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels
