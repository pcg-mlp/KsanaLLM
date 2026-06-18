/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include <cuda_runtime.h>

namespace ksana_llm {

// ===========================================================================
// Per-kernel timing statistics for MoE Forward profiling.
// Attached to a layer via SetPerfStats(); Forward writes CUDA-event-measured
// kernel times into this struct.  Callers Reset() between profiling windows.
//
// This is a standalone test-utility header — NOT part of the production layer
// hierarchy.  It is only included in *_test.cpp and in the concrete MoE layer
// headers that opt into profiling support.
// ===========================================================================
struct MoePerfStats {
  // Aggregated times in milliseconds for the profiling window since last Reset().
  float gemm1_time_ms = 0.f;
  float gemm2_time_ms = 0.f;
  float total_time_ms = 0.f;

  // FLOP counts for the current (cfg, num_tokens) combination, filled by the
  // benchmark driver so TFLOPS can be printed directly.
  double gemm1_flops = 0.;
  double gemm2_flops = 0.;
  double total_flops = 0.;

  void Reset() {
    gemm1_time_ms = 0.f;
    gemm2_time_ms = 0.f;
    total_time_ms = 0.f;
    gemm1_flops = 0.;
    gemm2_flops = 0.;
    total_flops = 0.;
  }

  double Gemm1TFLOPS() const { return gemm1_time_ms > 0.f ? (gemm1_flops / (gemm1_time_ms / 1e3)) / 1e12 : 0.; }
  double Gemm2TFLOPS() const { return gemm2_time_ms > 0.f ? (gemm2_flops / (gemm2_time_ms / 1e3)) / 1e12 : 0.; }
  double TotalTFLOPS() const { return total_time_ms > 0.f ? (total_flops / (total_time_ms / 1e3)) / 1e12 : 0.; }
};

}  // namespace ksana_llm
