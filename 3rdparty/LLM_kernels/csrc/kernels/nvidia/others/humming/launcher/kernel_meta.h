/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Kernel metadata for humming JIT-compiled kernels.
 */

#pragma once

#include <cuda.h>
#include <climits>
#include <cstdint>
#include <sstream>
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

// Human-readable summary of a KernelMeta, matching the format used by the test
// driver's PrintSelectedKernels table: "8x128x256 S3 C1 [SK]".
// Appended " [SK]" only when use_stream_k is true. Intended for logs and test
// output; not used on any hot path.
inline std::string FormatKernelMeta(const KernelMeta& m) {
  std::ostringstream oss;
  oss << m.block_m << "x" << m.block_n << "x" << m.block_k << " S" << m.num_stages << " C" << m.num_ctas_per_sm;
  if (m.use_stream_k) {
    oss << " [SK]";
  }
  return oss.str();
}

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
