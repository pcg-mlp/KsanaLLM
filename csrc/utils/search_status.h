// Copyright 2025 Tencent Inc.  All rights reserved.
#pragma once

#include <fmt/format.h>
#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration — KernelMeta is defined in humming launcher headers.
// Declared at global scope (NOT inside namespace ksana_llm) so it resolves to
// ::llm_kernels::nvidia::humming::KernelMeta. Placing it inside ksana_llm would
// create ksana_llm::llm_kernels, shadowing the global llm_kernels namespace and
// breaking all existing references to llm_kernels::nvidia::... within ksana_llm.
namespace llm_kernels {
namespace nvidia {
namespace humming {
struct KernelMeta;
}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels

namespace ksana_llm {

class CutlassSearchStatus {
 public:
  bool IsCutlassScheduleContain(size_t n, size_t k) {
    auto it1 = cutlass_schedule_cache.find(n);
    if (it1 != cutlass_schedule_cache.end()) {
      auto it2 = it1->second.find(k);
      if (it2 != it1->second.end()) {
        return true;
      }
    }
    return false;
  }

  bool IsCutlassWorkspaceContain(size_t m, size_t n, size_t k) {
    std::string index_str = fmt::format("({},{},{})", m, n, k);
    auto it = cutlass_workspace_cache.find(index_str);
    if (it != cutlass_workspace_cache.end()) {
      return true;
    }
    return false;
  }

  void AddCutlassSchedule(size_t n, size_t k, std::vector<size_t> best) { cutlass_schedule_cache[n][k] = best; }

  void AddCutlassWorkspace(size_t m, size_t n, size_t k, size_t ws) {
    std::string index_str = fmt::format("({},{},{})", m, n, k);
    cutlass_workspace_cache[index_str] = ws;
  }

  std::vector<size_t>& GetCutlassSchedule(size_t n, size_t k) { return cutlass_schedule_cache[n][k]; }

  size_t GetCutlassWorkspace(size_t m, size_t n, size_t k) {
    std::string index_str = fmt::format("({},{},{})", m, n, k);
    return cutlass_workspace_cache[index_str];
  }

  void ClearCutlassSchedule() { cutlass_schedule_cache.clear(); }

  void ClearCutlassWorkspace() { cutlass_workspace_cache.clear(); }

 private:
  std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t>>> cutlass_schedule_cache;
  std::unordered_map<std::string, size_t> cutlass_workspace_cache;
};

class MacheteSearchStatus {
 public:
  bool IsMacheteScheduleContain(size_t n, size_t k) {
    auto it1 = machete_schedule_cache.find(n);
    if (it1 != machete_schedule_cache.end()) {
      auto it2 = it1->second.find(k);
      if (it2 != it1->second.end()) {
        return true;
      }
    }
    return false;
  }

  bool IsMacheteWorkspaceContain(size_t m, size_t n, size_t k) {
    std::string index_str = fmt::format("({},{},{})", m, n, k);
    auto it = machete_workspace_cache.find(index_str);
    if (it != machete_workspace_cache.end()) {
      return true;
    }
    return false;
  }

  void AddMacheteSchedule(size_t n, size_t k, std::vector<std::string> best) { machete_schedule_cache[n][k] = best; }

  void AddMacheteWorkspace(size_t m, size_t n, size_t k, size_t ws) {
    std::string index_str = fmt::format("({},{},{})", m, n, k);
    machete_workspace_cache[index_str] = ws;
  }

  std::vector<std::string>& GetMacheteSchedule(size_t n, size_t k) { return machete_schedule_cache[n][k]; }

  size_t GetMacheteWorkspace(size_t m, size_t n, size_t k) {
    std::string index_str = fmt::format("({},{},{})", m, n, k);
    return machete_workspace_cache[index_str];
  }

  void ClearMacheteSchedule() { machete_schedule_cache.clear(); }

  void ClearMacheteWorkspace() { machete_workspace_cache.clear(); }

 private:
  std::unordered_map<size_t, std::unordered_map<size_t, std::vector<std::string>>> machete_schedule_cache;
  std::unordered_map<std::string, size_t> machete_workspace_cache;
};

class CutlassMoeSearchStatus {
 public:
  bool IsCutlassMoeScheduleContain(size_t s1, size_t s2, size_t s3, size_t s4) {
    std::string key = fmt::format("({},{},{},{})", s1, s2, s3, s4);
    auto it = cutlass_moe_schedule_cache.find(key);
    return it != cutlass_moe_schedule_cache.end();
  }

  void AddCutlassMoeSchedule(size_t s1, size_t s2, size_t s3, size_t s4, std::vector<std::vector<int64_t>> val) {
    std::string key = fmt::format("({},{},{},{})", s1, s2, s3, s4);
    cutlass_moe_schedule_cache[key] = val;
  }

  std::vector<std::vector<int64_t>>& GetCutlassMoeSchedule(size_t s1, size_t s2, size_t s3, size_t s4) {
    std::string key = fmt::format("({},{},{},{})", s1, s2, s3, s4);
    auto it = cutlass_moe_schedule_cache.find(key);
    return it->second;
  }

  void ClearCutlassMoeSchedule() { cutlass_moe_schedule_cache.clear(); }

 private:
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> cutlass_moe_schedule_cache;
};

class HummingMoeSearchStatus {
 public:
  // A profiled schedule: parallel arrays of size_m → best KernelMeta*.
  // size_ms is sorted ascending. To add search points, just push to both vectors.
  struct ProfiledSchedule {
    std::vector<int64_t> size_ms;
    std::vector<const llm_kernels::nvidia::humming::KernelMeta*> metas;
  };

  bool IsProfiled(int64_t n, int64_t k, int64_t e, int dt, int gt) const {
    std::string key = fmt::format("({},{},{},{},{})", n, k, e, dt, gt);
    return humming_moe_schedule_cache_.find(key) != humming_moe_schedule_cache_.end();
  }

  void AddProfiledSchedule(int64_t n, int64_t k, int64_t e, int dt, int gt, const ProfiledSchedule& schedule) {
    std::string key = fmt::format("({},{},{},{},{})", n, k, e, dt, gt);
    humming_moe_schedule_cache_[key] = schedule;
  }

  // Right-align lookup: returns the KernelMeta* for the smallest profiled size_m >= requested.
  // If requested > max profiled, returns the largest profiled entry (large-M trend is consistent).
  // Returns nullptr if this (N,K,E,dt,gt) has not been profiled.
  const llm_kernels::nvidia::humming::KernelMeta* GetProfiledMeta(int64_t n, int64_t k, int64_t e, int dt, int gt,
                                                                  int64_t size_m) const {
    std::string key = fmt::format("({},{},{},{},{})", n, k, e, dt, gt);
    auto it = humming_moe_schedule_cache_.find(key);
    if (it == humming_moe_schedule_cache_.end()) return nullptr;
    const auto& sched = it->second;
    if (sched.size_ms.empty()) return nullptr;
    // Binary search: first size_m >= requested
    auto idx_it = std::lower_bound(sched.size_ms.begin(), sched.size_ms.end(), size_m);
    if (idx_it != sched.size_ms.end()) {
      return sched.metas[std::distance(sched.size_ms.begin(), idx_it)];
    }
    return sched.metas.back();  // requested > max → use largest
  }

  void ClearHummingMoeSchedule() { humming_moe_schedule_cache_.clear(); }

 private:
  std::unordered_map<std::string, ProfiledSchedule> humming_moe_schedule_cache_;
};

}  // namespace ksana_llm
