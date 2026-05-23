/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Utility functions for the Marlin MoE WNA16 operator.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "scalar_type.hpp"

namespace llm_kernels {
namespace nvidia {
namespace marlin_moe_wna16 {

// Constants re-exported for callers
static constexpr int kTileSize = 16;
static constexpr int kMinThreadN = 64;
static constexpr int kMaxThreadN = 256;

/// Calculate minimum workspace size in number of int elements.
int64_t get_workspace_size(int64_t size_n, int64_t sorted_token_ids_len, int64_t moe_block_size);

/// Calculate c_tmp buffer size in number of float elements.
/// Only needed when use_fp32_reduce=true && use_atomic_add=false.
int64_t get_c_tmp_size(int64_t size_n, int64_t sorted_token_ids_len,
                       int64_t moe_block_size, ::marlin_moe_wna16::ScalarTypeId b_q_type_id);

/// Calculate a_tmp buffer size in number of compute-type elements.
/// Only needed when has_act_order=true.
int64_t get_a_tmp_size(int64_t size_m, int64_t size_k, int64_t top_k);

}  // namespace marlin_moe_wna16
}  // namespace nvidia
}  // namespace llm_kernels
