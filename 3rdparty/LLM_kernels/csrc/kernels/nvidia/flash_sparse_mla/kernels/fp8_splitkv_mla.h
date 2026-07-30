/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#pragma once

#include "params.h"

namespace llm_kernels {
namespace nvidia {

// A modified version of flash sparse mla that computes in FP8 precision
template <ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params);

template <ModelType MODEL_TYPE, int NUM_HEADS, int BLOCK_SIZE_N>
void run_flash_splitkv_mla_fp8_sparse_swapab_kernel(const SparseAttnDecodeParams &params);

}  // namespace nvidia
}  // namespace llm_kernels
