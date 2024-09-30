/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

void awq_marlin_repack(const uint32_t* b_q_weight_ptr, uint32_t* out_ptr, int64_t size_k, int64_t size_n,
                       int64_t num_bits, int rank, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels