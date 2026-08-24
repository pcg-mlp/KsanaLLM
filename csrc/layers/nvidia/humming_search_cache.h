/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "csrc/kernels/nvidia/others/humming/launcher/kernel_meta.h"
#include "csrc/utils/search_status.h"

namespace ksana_llm {

// Parameters identifying one humming MoE GEMM search problem.
// Key-identity fields (n/k/e/dt/gt/top_k/size_m_multiplier/ep_size/using_deepep/
// gpu_name/sm_count + the candidate kernel set passed alongside) determine the
// cache key.
// The remaining fields are diagnostics stored in the file for human inspection only.
struct HummingSearchCacheKeyParams {
  // Key identity
  int64_t n = 0;
  int64_t k = 0;
  int64_t e = 0;
  int dt = 0;  // llm_kernels::nvidia::humming::ComputeDtype as int
  int gt = 0;  // llm_kernels::nvidia::humming::GemmType as int
  int64_t top_k = 1;
  int64_t size_m_multiplier = 1;
  int64_t ep_size = 1;  // global_expert_para_size_
  bool using_deepep = false;
  // Hardware identity (heuristic class is chosen by GPU name; stream-K scheduling
  // scales with SM count — same-name-candidates GPUs with different SM counts must
  // not share a cache entry)
  std::string gpu_name;
  int sm_count = 0;
  // Diagnostics (not part of key)
  int rank = 0;
  size_t profile_iters = 0;
  size_t warmup_iters = 0;
  size_t m_thresh1 = 0;
  std::string cc;
};

// File-level cross-process cache for humming MoE operator-search results.
// Pure logic, no CUDA runtime dependency (candidates passed in by caller, e.g.
// from GetHummingKernelCandidates). Any failure degrades to a miss / skipped
// save — never affects correctness.
class HummingMoeSearchCache {
 public:
  using KernelCandidates = std::vector<const llm_kernels::nvidia::humming::KernelMeta*>;

  // Bump when search logic, measurement method, or bucket list changes.
  static constexpr int kSearchCacheFormatVersion = 1;

  // cache_dir empty → $KSANA_HUMMING_KERNEL_PATH (or ~/.cache/KsanaLLM/humming_kernel_files)
  // + "/humming_search_cache". Explicit dir is used by unit tests.
  explicit HummingMoeSearchCache(const std::string& cache_dir = "");

  // Deterministic cache key: sha256 hex over canonical key material built from
  // key-identity params plus the sorted kernel_name list of candidates.
  // Candidates must be the filtered set for THIS problem (see
  // GetHummingKernelCandidates) — this class does not filter by shape itself.
  std::string ComputeCacheKey(const HummingSearchCacheKeyParams& params, const KernelCandidates& candidates) const;

  // File path: {cache_dir}/humming_search_{key16}_n{n}_k{k}_rank{rank}.json
  std::string CacheFilePath(const HummingSearchCacheKeyParams& params, const KernelCandidates& candidates) const;

  // Load + full validation chain. On success fills out_schedule (metas resolved
  // to pointers from candidates — valid as long as the underlying kernel table lives).
  // Returns false on any failure; never throws.
  bool TryLoad(const HummingSearchCacheKeyParams& params, const KernelCandidates& candidates,
               HummingMoeSearchStatus::ProfiledSchedule& out_schedule) const;

  // Atomically write the cache file ({path}.tmp.{pid} + rename).
  // Returns false on any failure (disabled / unwritable dir / io error).
  bool Save(const HummingSearchCacheKeyParams& params, const KernelCandidates& candidates,
            const HummingMoeSearchStatus::ProfiledSchedule& schedule) const;

  // HUMMING_SEARCH_CACHE=0 → TryLoad and Save are no-ops. Read per call (not cached).
  static bool Enabled();
  // HUMMING_SEARCH_REGENERATE=1 → TryLoad is a no-op (always re-search); Save overwrites.
  static bool Regenerate();

  // SHA256 of a byte string as 64-char lowercase hex. Public for tests.
  static std::string Sha256Hex(const std::string& data);

 private:
  std::string cache_dir_;
};

}  // namespace ksana_llm
