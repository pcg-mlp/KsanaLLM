/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Kernel metadata for humming JIT-compiled kernels.
 */

#pragma once

#include <cuda.h>
#include <climits>
#include <cstdint>
#include <string>

namespace llm_kernels {
namespace nvidia {
namespace humming {

/// MoE GEMM dispatch type
enum class GemmType : int {
  INDEXED = 1,
  GROUPED_CONTIGUOUS = 2,
};

/// Compute dtype selection
enum class ComputeDtype : int {
  BF16 = 0,
  FP8 = 1,
};

/// Metadata for a single JIT-compiled kernel variant.
struct KernelMeta {
  CUfunction func;
  std::string kernel_name;  // Mangled kernel symbol name from ELF
  int smem_size;
  int num_threads;
  int64_t size_n;
  int64_t size_k;
  int block_m, block_n, block_k;
  int num_stages;
  int num_ctas_per_sm;
  bool use_tma_a;
  bool use_tma_b;
  bool use_tma_c;
  bool use_tma_bs;
  bool use_stream_k;
  bool use_warp_spec;
  bool use_fused_e8m0_scale;
  GemmType gemm_type;
  ComputeDtype compute_dtype;
  // Range of shape_m values for which this kernel is optimal (from heuristic)
  int64_t min_shape_m = 1;
  int64_t max_shape_m = INT64_MAX;
};

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
