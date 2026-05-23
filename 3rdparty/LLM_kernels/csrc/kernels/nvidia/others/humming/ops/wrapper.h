/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Workspace sizing helpers for humming MoE GEMM.
 */

#pragma once

#include <cstdint>

namespace llm_kernels {
namespace nvidia {
namespace humming {

enum class ComputeDtype : int;

/// Calculate locks buffer size in int32 elements.
int64_t get_locks_size(int64_t size_m, int64_t size_n, int64_t size_k,
                       ComputeDtype compute_dtype);

/// Calculate total workspace size in bytes.
int64_t get_workspace_size(int64_t size_m, int64_t size_n, int64_t size_k,
                           int64_t num_experts, int64_t top_k,
                           ComputeDtype compute_dtype);

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
