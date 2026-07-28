/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include <memory>
#include <mutex>

#include "csrc/kernels/nvidia/others/humming/jit/humming_jit_compiler.h"

namespace ksana_llm {

// Process-global shared JIT compiler instance for all FP4 Humming MoE layers.
// W4A16 and W4A8 kernels compile into the SAME compiler table (the ops dispatch
// layer stores one global compiler via SetJitCompiler); separate instances would
// clobber each other.  This header is precision-agnostic infrastructure — the
// only shared piece between HummingFusedFp4A16MoeLayer and HummingFusedFp4A8MoeLayer.
inline std::shared_ptr<llm_kernels::nvidia::humming::HummingJitCompiler>& GetHummingFp4JitCompiler() {
  static std::shared_ptr<llm_kernels::nvidia::humming::HummingJitCompiler> instance;
  return instance;
}

inline std::mutex& GetHummingFp4JitMutex() {
  static std::mutex mtx;
  return mtx;
}

}  // namespace ksana_llm
