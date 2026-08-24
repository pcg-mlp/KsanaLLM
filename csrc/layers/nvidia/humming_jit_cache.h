/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "csrc/kernels/nvidia/others/humming/jit/humming_jit_compiler.h"

namespace ksana_llm {

// Process-global shared JIT compiler instance for all FP4 Humming MoE layers.
// W4A16 and W4A8 kernels compile into the SAME compiler table (the ops dispatch
// layer stores one global compiler via SetJitCompiler); separate instances would
// clobber each other.  This header is precision-agnostic infrastructure — the
// only shared piece between HummingFusedWfp4A16MoeLayer and HummingFusedWfp4Afp8MoeLayer.
inline std::shared_ptr<llm_kernels::nvidia::humming::HummingJitCompiler>& GetHummingFp4JitCompiler() {
  static std::shared_ptr<llm_kernels::nvidia::humming::HummingJitCompiler> instance;
  return instance;
}

inline std::mutex& GetHummingFp4JitMutex() {
  static std::mutex mtx;
  return mtx;
}

// Filtered candidate set for one humming MoE GEMM problem: kernels matching
// (size_n, size_k, gemm_type, compute_dtype). Single source of truth for the
// candidate filtering that profiling and the search-cache key both depend on —
// consumers must not re-implement the match inline. Returns an empty vector
// when the JIT compiler is not initialized.
//
// NOTE: for workspace sizing use the raw GetJitKernelTable() scan instead —
// buffers must cover the worst case across ALL compiled shapes (max block_m /
// num_ctas_per_sm), not just one problem's candidates.
std::vector<const llm_kernels::nvidia::humming::KernelMeta*> GetHummingKernelCandidates(
    int64_t size_n, int64_t size_k, llm_kernels::nvidia::humming::ComputeDtype compute_dtype,
    llm_kernels::nvidia::humming::GemmType gemm_type);

// Safe access to the FULL JIT kernel table (nullptr when the compiler is not
// initialized). Used for cross-shape aggregation such as workspace sizing —
// intentionally unfiltered by (n, k); see GetHummingKernelCandidates note.
const std::vector<llm_kernels::nvidia::humming::KernelMeta>* GetJitKernelTable();

}  // namespace ksana_llm
