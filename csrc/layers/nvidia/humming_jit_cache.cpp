/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/nvidia/humming_jit_cache.h"

namespace ksana_llm {

std::vector<const llm_kernels::nvidia::humming::KernelMeta*> GetHummingKernelCandidates(
    int64_t size_n, int64_t size_k, llm_kernels::nvidia::humming::ComputeDtype compute_dtype,
    llm_kernels::nvidia::humming::GemmType gemm_type) {
  std::vector<const llm_kernels::nvidia::humming::KernelMeta*> candidates;
  auto& compiler = GetHummingFp4JitCompiler();
  if (!compiler) return candidates;
  for (const auto& meta : compiler->GetKernelTable()) {
    if (meta.size_n != size_n || meta.size_k != size_k) continue;
    if (meta.gemm_type != gemm_type) continue;
    if (meta.compute_dtype != compute_dtype) continue;
    candidates.push_back(&meta);
  }
  return candidates;
}

const std::vector<llm_kernels::nvidia::humming::KernelMeta>* GetJitKernelTable() {
  auto& compiler = GetHummingFp4JitCompiler();
  if (!compiler) return nullptr;
  return &compiler->GetKernelTable();
}

}  // namespace ksana_llm
