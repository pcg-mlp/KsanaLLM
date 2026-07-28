/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

// FP4 MoE test — configuration types and constants: backend enum/spec type,
// model configs, token sizes, benchmark iteration constants, and shared
// derived-dimension helpers.
//
// Backend metadata table (MasterBackendSpecs) and FP4_BACKENDS env parsing
// (ParseRequestedBackends) live in common.h — they are runtime selection
// logic used only by the driver, not configuration.
//
// Expected to be included after the layer/kernel/torch headers from
// fp4_moe_layer_test.cpp's ENABLE_CUDA block (uses
// llm_kernels::nvidia::humming::KernelMeta in BackendSpec).

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

namespace ksana_llm {

// ===========================================================================
// Backend enum
// ===========================================================================
enum class Fp4Backend { kTriton = 0, kMarlin = 1, kHummingW4A16 = 2, kHummingW4A8 = 3 };

// ===========================================================================
// Backend descriptor — single source of truth for backend metadata.
// Drives the run loop, all print tables, and accuracy verification.
// ===========================================================================
struct BackendSpec {
  Fp4Backend backend;
  const char* name;      // display name for table headers
  const char* env_name;  // canonical name for FP4_BACKENDS matching
  const char* alias;     // short alias for FP4_BACKENDS matching (may be null)
  float cos_threshold;   // accuracy cosine-similarity threshold
  bool is_humming;       // true for humming-only tables
  bool needs_fp8_ref;    // true for W4A8 (selects FP8 GPU reference)
};

// ===========================================================================
// Model configuration
// ===========================================================================
struct Fp4MoeTestConfig {
  std::string name;
  size_t hidden_size;
  size_t inter_size;
  size_t num_experts;
  size_t tp_size;
  size_t expert_para_size;
};

static const Fp4MoeTestConfig kFlashTP4 = {"Flash_TP4", 4096, 2048, 256, 4, 1};
static const Fp4MoeTestConfig kFlashEP4 = {"Flash_EP4", 4096, 2048, 256, 1, 4};
static const Fp4MoeTestConfig kProTP8 = {"Pro_TP8", 7168, 3072, 384, 8, 1};
static const Fp4MoeTestConfig kProEP8 = {"Pro_EP8", 7168, 3072, 384, 1, 8};

// ===========================================================================
// Token sizes for benchmarking
// NOTE(jinxcwu) 可以多加一点做完整的性能测试覆盖，目前为了节省测试时间设置的比较小
// ===========================================================================
static const std::vector<size_t> kTokenSizes = {1, 4, 16};
// static const std::vector<size_t> kTokenSizes = {1,   2,   4,    8,    16,   32,   64,    128,
//                                                 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

static constexpr int kWarmupIters = 20;
static constexpr int kBenchIters = 50;

// ===========================================================================
// Derived dimensions
// ===========================================================================
inline size_t ComputeMoeTp(const Fp4MoeTestConfig& cfg) {
  return std::max(size_t(1), cfg.tp_size / cfg.expert_para_size);
}
inline size_t ComputeInterPerRank(const Fp4MoeTestConfig& cfg) {
  size_t moe_tp = ComputeMoeTp(cfg);
  return (cfg.inter_size + moe_tp - 1) / moe_tp;
}
inline size_t ComputeExpertsPerRank(const Fp4MoeTestConfig& cfg) {
  return (cfg.num_experts + cfg.expert_para_size - 1) / cfg.expert_para_size;
}

}  // namespace ksana_llm
