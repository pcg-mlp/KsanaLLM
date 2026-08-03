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
// wfp4_moe_layer_test.cpp's ENABLE_CUDA block (uses
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
// Model type — identifies which family of model a config targets.
// Each type fixes the model-level dimensions (hidden/inter/experts/topk) and
// the scoring function, so per-config instances only need to specify the
// parallelism strategy (tp_size / expert_para_size).
// ===========================================================================
enum class ModelType { kV4Flash = 0, kV4Pro = 1, kK3 = 2 };

struct ModelSpec {
  ModelType type;
  const char* name;  // short tag used in config names
  size_t hidden_size;
  size_t inter_size;
  size_t num_experts;
  size_t topk;
  const char* scoring_func;     // "sqrtsoftplus" (DSv4) or "sigmoid" (K3)
  float swiglu_limit;           // SwiGLU clamp limit (DSv4: 10.0, K3: 0.0 = unused)
  float situ_beta;              // SiTU activation beta (K3: 4.0, DSv4: 0.0 = unused)
  float situ_linear_beta;       // SiTU linear beta (K3: 25.0, DSv4: 0.0 = unused)
  float routed_scaling_factor;  // routing weight scale (DSv4: 1.5, K3: 1.0)
};

// Fixed model-level parameters for each model family.
inline const ModelSpec& GetModelSpec(ModelType type) {
  static const ModelSpec kSpecs[] = {
      // V4 Flash: DSv4 flash model — sqrtsoftplus scoring + SwiGLU (limit=10).
      {ModelType::kV4Flash, "v4flash", 4096, 2048, 256, 6, "sqrtsoftplus", 10.0f, 0.0f, 0.0f, 1.5f},
      // V4 Pro: DSv4 pro model — sqrtsoftplus scoring + SwiGLU (limit=10).
      {ModelType::kV4Pro, "v4pro", 7168, 3072, 384, 6, "sqrtsoftplus", 10.0f, 0.0f, 0.0f, 1.5f},
      // K3: latent dim 3584 (NOT model hidden 7168), sigmoid scoring + SiTU
      // activation (situ_beta=4.0, situ_linear_beta=25.0), topk=16.
      {ModelType::kK3, "k3", 3584, 3072, 896, 16, "sigmoid", 0.0f, 4.0f, 25.0f, 1.0f},
  };
  return kSpecs[static_cast<size_t>(type)];
}

inline bool IsK3(ModelType type) { return type == ModelType::kK3; }

// ===========================================================================
// Model configuration
// ===========================================================================
struct Fp4MoeTestConfig {
  std::string name;
  ModelType model_type;  // fixes hidden/inter/experts/topk/scoring
  size_t tp_size;
  size_t expert_para_size;

  // Accessors delegating to the fixed ModelSpec.
  const ModelSpec& spec() const { return GetModelSpec(model_type); }
  size_t hidden_size() const { return spec().hidden_size; }
  size_t inter_size() const { return spec().inter_size; }
  size_t num_experts() const { return spec().num_experts; }
  size_t topk() const { return spec().topk; }
  const char* scoring_func() const { return spec().scoring_func; }
  float swiglu_limit() const { return spec().swiglu_limit; }
  float situ_beta() const { return spec().situ_beta; }
  float situ_linear_beta() const { return spec().situ_linear_beta; }
  float routed_scaling_factor() const { return spec().routed_scaling_factor; }
  bool is_k3() const { return IsK3(model_type); }
};

static const Fp4MoeTestConfig kV4FlashTP4 = {"v4flash_TP4", ModelType::kV4Flash, 4, 1};
static const Fp4MoeTestConfig kV4FlashEP4 = {"v4flash_EP4", ModelType::kV4Flash, 1, 4};
static const Fp4MoeTestConfig kV4ProTP8 = {"v4pro_TP8", ModelType::kV4Pro, 8, 1};
static const Fp4MoeTestConfig kV4ProEP8 = {"v4pro_EP8", ModelType::kV4Pro, 1, 8};
// K3 TP8: no EP. hidden_size=3584 is the LatentMoE latent dim (NOT model hidden 7168).
static const Fp4MoeTestConfig kK3TP8 = {"k3_TP8", ModelType::kK3, 8, 1};

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
  return (cfg.inter_size() + moe_tp - 1) / moe_tp;
}
inline size_t ComputeExpertsPerRank(const Fp4MoeTestConfig& cfg) {
  return (cfg.num_experts() + cfg.expert_para_size - 1) / cfg.expert_para_size;
}

}  // namespace ksana_llm
