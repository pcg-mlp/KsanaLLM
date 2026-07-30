/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace llm_kernels {
namespace nvidia {
namespace test {

enum class ModelType { kV32, kMODEL1 };

struct ModelConfig {
  ModelType type;
  const char* name;
  size_t head_size_k;
  size_t head_size_v;
  size_t kcache_batch_stride;
  size_t topk;

  bool IsModel1() const { return type == ModelType::kMODEL1; }
};

static constexpr size_t kBlockSize = 64;
static constexpr size_t kCacheBlockPadding = 132;

static const ModelConfig kModelConfigs[] = {
    {ModelType::kV32, "V32",
     /*head_size_k=*/576, /*head_size_v=*/512,
     /*kcache_batch_stride=*/kBlockSize * (656 + kCacheBlockPadding),
     /*topk=*/2048},
    {ModelType::kMODEL1, "MODEL1_topk512",
     /*head_size_k=*/512, /*head_size_v=*/512,
     /*kcache_batch_stride=*/kBlockSize * (584 + kCacheBlockPadding),
     /*topk=*/512},
    {ModelType::kMODEL1, "MODEL1_topk1024",
     /*head_size_k=*/512, /*head_size_v=*/512,
     /*kcache_batch_stride=*/kBlockSize * (584 + kCacheBlockPadding),
     /*topk=*/1024},
};

enum class MlaBackend { kFp8DsMla, kFp8E4M3, kFp8E4M3SwapAB };

struct MlaBackendSpec {
  MlaBackend backend;
  const char* name;
  bool needs_q_padding;
};

inline const std::vector<MlaBackendSpec>& MasterMlaBackends() {
  static const std::vector<MlaBackendSpec> kMaster = {
      {MlaBackend::kFp8DsMla, "fp8_ds_mla", true},
      {MlaBackend::kFp8E4M3, "fp8_e4m3", false},
      {MlaBackend::kFp8E4M3SwapAB, "fp8_e4m3_swapab", false},
  };
  return kMaster;
}

inline std::vector<MlaBackendSpec> ParseMlaBackends() {
  const char* env = std::getenv("MLA_BACKENDS");
  if (env == nullptr || env[0] == '\0') {
    return MasterMlaBackends();
  }

  auto match = [](const std::string& token, const MlaBackendSpec& spec) {
    std::string t = token;
    t.erase(std::remove_if(t.begin(), t.end(), ::isspace), t.end());
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    return t == spec.name;
  };

  std::vector<MlaBackendSpec> selected;
  std::string s(env);
  size_t start = 0;
  while (start <= s.size()) {
    size_t comma = s.find(',', start);
    std::string token = (comma == std::string::npos) ? s.substr(start) : s.substr(start, comma - start);
    bool found = false;
    std::string trimmed = token;
    trimmed.erase(std::remove_if(trimmed.begin(), trimmed.end(), ::isspace), trimmed.end());
    for (const auto& spec : MasterMlaBackends()) {
      if (match(trimmed, spec)) {
        bool already = false;
        for (const auto& sel : selected) {
          if (sel.backend == spec.backend) {
            already = true;
            break;
          }
        }
        if (!already) {
          selected.push_back(spec);
        }
        found = true;
        break;
      }
    }
    if (!found) {
      std::cerr << "MLA_BACKENDS: unknown backend '" << trimmed << "', skipping" << std::endl;
    }
    if (comma == std::string::npos) break;
    start = comma + 1;
  }

  if (selected.empty()) {
    std::cerr << "MLA_BACKENDS: no valid backends parsed, falling back to all" << std::endl;
    return MasterMlaBackends();
  }

  std::vector<MlaBackendSpec> ordered;
  ordered.reserve(selected.size());
  for (const auto& master_spec : MasterMlaBackends()) {
    for (const auto& sel_spec : selected) {
      if (sel_spec.backend == master_spec.backend) {
        ordered.push_back(sel_spec);
        break;
      }
    }
  }
  selected = std::move(ordered);

  return selected;
}

enum class IndicesStrategy { kSequential, kRandom, kClustered };

struct MlaTestRange {
  std::vector<size_t> batch_sizes;
  std::vector<size_t> h_q_cases;
  std::vector<size_t> seqlen_q_cases;
  std::vector<size_t> seqlen_ks;
  std::vector<IndicesStrategy> idx_strategies;
  int warmups = 5;
  int iterations = 10;
  bool run_accuracy = true;
  bool run_perf = true;
};

// Full (dev) range
static const MlaTestRange kFullRange = {
    /*batch_sizes=*/{1, 2, 8, 32, 64, 128},
    /*h_q_cases=*/{16, 64, 128},
    /*seqlen_q_cases=*/{1, 3, 6},
    /*seqlen_ks=*/{512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},
    /*idx_strategies=*/{IndicesStrategy::kSequential, IndicesStrategy::kRandom, IndicesStrategy::kClustered},
    /*warmups=*/5,
    /*iterations=*/10,
    /*run_accuracy=*/true,
    /*run_perf=*/true,
};

// Full (dev) range
static const MlaTestRange kFastRange = {
    /*batch_sizes=*/{16, 32, 64},
    /*h_q_cases=*/{16},
    /*seqlen_q_cases=*/{1, 3, 6},
    /*seqlen_ks=*/{1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},
    /*idx_strategies=*/{IndicesStrategy::kSequential, IndicesStrategy::kRandom, IndicesStrategy::kClustered},
    /*warmups=*/5,
    /*iterations=*/10,
    /*run_accuracy=*/true,
    /*run_perf=*/true,
};

// Compact (CI/CD) range
static const MlaTestRange kCICIRange = {
    /*batch_sizes=*/{1, 128},
    /*h_q_cases=*/{16},
    /*seqlen_q_cases=*/{1},
    /*seqlen_ks=*/{512, 8192, 131072},
    /*idx_strategies=*/{IndicesStrategy::kSequential},
    /*warmups=*/2,
    /*iterations=*/5,
    /*run_accuracy=*/true,
    /*run_perf=*/true,
};

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
