/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include <cuda_runtime.h>

// Forward declaration — KernelMeta is defined in humming launcher headers.
// Declared at global scope (NOT inside namespace ksana_llm) so it resolves to
// ::llm_kernels::nvidia::humming::KernelMeta. Mirrors the pattern in
// csrc/utils/search_status.h.
namespace llm_kernels {
namespace nvidia {
namespace humming {
struct KernelMeta;
}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels

namespace ksana_llm {

// ===========================================================================
// Per-kernel timing statistics for MoE Forward profiling.
// Attached to a layer via SetPerfStats(); Forward writes CUDA-event-measured
// kernel times into this struct.  Callers Reset() between profiling windows.
//
// This is a standalone test-utility header — NOT part of the production layer
// hierarchy.  It is only included in *_test.cpp and in the concrete MoE layer
// headers that opt into profiling support.
//
// Fields marked "humming-only" are left at 0/nullptr by MarlinFp4MoeLayer.
// ===========================================================================
struct MoePerfStats {
  // Aggregated times in milliseconds for the profiling window since last Reset().
  float gemm1_time_ms = 0.f;
  float gemm2_time_ms = 0.f;
  float total_time_ms = 0.f;

  // Humming-only detailed non-GEMM breakdown (0 for marlin).
  // align = GEMM1 align + GEMM2 re-align; quant = GEMM1 quant + GEMM2 quant.
  float gating_time_ms = 0.f;   // TopK softplus-sqrt
  float align_time_ms = 0.f;    // MoE align (GEMM1 + GEMM2)
  float quant_time_ms = 0.f;    // FP8 activation quant (W4A8 only; 0 for W4A16)
  float silu_time_ms = 0.f;     // SiluAndMul
  float mulsum_time_ms = 0.f;   // MoeFusedMulSum

  // Humming-only: last selected JIT kernel meta (nullptr for marlin).
  // Set once per ForwardT call (same meta across all benchmark iterations for
  // a given token size); read by the test driver after the bench loop.
  const llm_kernels::nvidia::humming::KernelMeta* last_gemm1_meta = nullptr;
  const llm_kernels::nvidia::humming::KernelMeta* last_gemm2_meta = nullptr;

  // FLOP counts for the current (cfg, num_tokens) combination, filled by the
  // benchmark driver so TFLOPS can be printed directly.
  double gemm1_flops = 0.;
  double gemm2_flops = 0.;
  double total_flops = 0.;

  void Reset() {
    gemm1_time_ms = 0.f;
    gemm2_time_ms = 0.f;
    total_time_ms = 0.f;
    gating_time_ms = 0.f;
    align_time_ms = 0.f;
    quant_time_ms = 0.f;
    silu_time_ms = 0.f;
    mulsum_time_ms = 0.f;
    last_gemm1_meta = nullptr;
    last_gemm2_meta = nullptr;
    gemm1_flops = 0.;
    gemm2_flops = 0.;
    total_flops = 0.;
  }

  double Gemm1TFLOPS() const { return gemm1_time_ms > 0.f ? (gemm1_flops / (gemm1_time_ms / 1e3)) / 1e12 : 0.; }
  double Gemm2TFLOPS() const { return gemm2_time_ms > 0.f ? (gemm2_flops / (gemm2_time_ms / 1e3)) / 1e12 : 0.; }
  double TotalTFLOPS() const { return total_time_ms > 0.f ? (total_flops / (total_time_ms / 1e3)) / 1e12 : 0.; }
};

}  // namespace ksana_llm
