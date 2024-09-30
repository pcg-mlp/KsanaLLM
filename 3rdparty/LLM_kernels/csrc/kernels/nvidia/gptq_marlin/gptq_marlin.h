/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

namespace marlin {
int determine_reduce_max_m(int prob_m, int max_par);
}  // namespace marlin

void gptq_marlin_gemm(void* a, void* a_tmp, void* b_q_weight, void* b_scales, void* b_zeros, void* g_idx, void* perm,
                      void* workspace, void* c, void* c_tmp, int64_t size_m, int64_t size_n, int64_t size_k,
                      int64_t num_groups, bool is_k_full, bool has_zp, bool has_act_order, bool is_awq, int rank,
                      cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels