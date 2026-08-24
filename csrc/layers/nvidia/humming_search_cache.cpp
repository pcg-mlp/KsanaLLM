/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/nvidia/humming_search_cache.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <unordered_map>

#include "csrc/kernels/nvidia/others/humming/jit/humming_jit_compiler.h"
#include "global_cache_connector/sha256_lite.h"
#include "nlohmann/json.hpp"
#include "utils/logger.h"

namespace ksana_llm {

namespace {

// Sorted candidate kernel_names (cache key material — order-independent).
std::vector<std::string> CandidateNames(const HummingMoeSearchCache::KernelCandidates& candidates) {
  std::vector<std::string> names;
  names.reserve(candidates.size());
  for (const auto* meta : candidates) names.push_back(meta->kernel_name);
  std::sort(names.begin(), names.end());
  return names;
}

std::string JoinNames(const std::vector<std::string>& names) {
  std::string joined;
  for (const auto& n : names) {
    if (!joined.empty()) joined += ",";
    joined += n;
  }
  return joined;
}

std::string IsoNowUtc() {
  std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

nlohmann::json ParamsToJson(const HummingSearchCacheKeyParams& p, const std::vector<std::string>& kernel_names) {
  return nlohmann::json{{"n", p.n},
                        {"k", p.k},
                        {"E", p.e},
                        {"dt", p.dt},
                        {"gt", p.gt},
                        {"top_k", p.top_k},
                        {"size_m_multiplier", p.size_m_multiplier},
                        {"ep_size", p.ep_size},
                        {"using_deepep", p.using_deepep},
                        {"rank", p.rank},
                        {"profile_iters", p.profile_iters},
                        {"warmup_iters", p.warmup_iters},
                        {"m_thresh1", p.m_thresh1},
                        {"gpu_name", p.gpu_name},
                        {"sm_count", p.sm_count},
                        {"cc", p.cc},
                        {"kernel_names", kernel_names}};
}

nlohmann::json ScheduleToJson(const HummingMoeSearchStatus::ProfiledSchedule& schedule) {
  nlohmann::json arr = nlohmann::json::array();
  for (size_t i = 0; i < schedule.size_ms.size(); ++i) {
    arr.push_back({{"size_m", schedule.size_ms[i]}, {"kernel_name", schedule.metas[i]->kernel_name}});
  }
  return arr;
}

}  // namespace

HummingMoeSearchCache::HummingMoeSearchCache(const std::string& cache_dir) {
  if (!cache_dir.empty()) {
    cache_dir_ = cache_dir;
    return;
  }
  // Single source of truth: the JIT-side shared resolver (KSANA_HUMMING_KERNEL_PATH
  // or its default). Never parse the env here — divergent defaults would split
  // the cache namespace.
  cache_dir_ = llm_kernels::nvidia::humming::GetKernelCacheRootDir() + "/humming_search_cache";
}

std::string HummingMoeSearchCache::Sha256Hex(const std::string& data) {
  uint8_t digest[32];
  sha256_detail::Sha256(reinterpret_cast<const uint8_t*>(data.data()), data.size(), digest);
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (int i = 0; i < 32; ++i) {
    out.push_back(kHex[digest[i] >> 4]);
    out.push_back(kHex[digest[i] & 0xf]);
  }
  return out;
}

std::string HummingMoeSearchCache::ComputeCacheKey(const HummingSearchCacheKeyParams& params,
                                                   const KernelCandidates& candidates) const {
  const std::vector<std::string> names = CandidateNames(candidates);
  // gpu_name + sm_count are part of the key: the heuristic picks its class by GPU
  // name (H20 vs generic Sm90) and stream-K scheduling depends on SM count, so two
  // GPUs with identical candidate names but different SM counts must not share a
  // cache (e.g. H100 PCIe 114 SM vs H200 132 SM — same generic Sm90 heuristic).
  const std::string material =
      fmt::format("humming_search|n={}|k={}|e={}|dt={}|gt={}|topk={}|mult={}|ep={}|deepep={}|gpu={}|sm={}|kernels={}",
                  params.n, params.k, params.e, params.dt, params.gt, params.top_k, params.size_m_multiplier,
                  params.ep_size, params.using_deepep, params.gpu_name, params.sm_count, JoinNames(names));
  return Sha256Hex(material);
}

std::string HummingMoeSearchCache::CacheFilePath(const HummingSearchCacheKeyParams& params,
                                                 const KernelCandidates& candidates) const {
  return fmt::format("{}/humming_search_{}_n{}_k{}_rank{}.json", cache_dir_,
                     ComputeCacheKey(params, candidates).substr(0, 16), params.n, params.k, params.rank);
}

bool HummingMoeSearchCache::Enabled() {
  const char* v = std::getenv("HUMMING_SEARCH_CACHE");
  return !(v && std::string(v) == "0");
}

bool HummingMoeSearchCache::Regenerate() {
  const char* v = std::getenv("HUMMING_SEARCH_REGENERATE");
  return v && std::string(v) == "1";
}

bool HummingMoeSearchCache::Save(const HummingSearchCacheKeyParams& params, const KernelCandidates& candidates,
                                 const HummingMoeSearchStatus::ProfiledSchedule& schedule) const {
  if (!Enabled()) return false;
  std::error_code ec;
  std::filesystem::create_directories(cache_dir_, ec);
  if (ec) {
    KLLM_LOG_WARNING << fmt::format("HummingMoeSearchCache: cannot create dir '{}': {}", cache_dir_, ec.message());
    return false;
  }

  const std::vector<std::string> names = CandidateNames(candidates);
  const nlohmann::json payload =
      nlohmann::json{{"params", ParamsToJson(params, names)}, {"schedule", ScheduleToJson(schedule)}};
  const nlohmann::json doc{{"format_version", kSearchCacheFormatVersion},
                           {"cache_key", ComputeCacheKey(params, candidates)},
                           {"rank", params.rank},
                           {"created_at", IsoNowUtc()},
                           {"params", payload["params"]},
                           {"schedule", payload["schedule"]},
                           {"payload_sha256", Sha256Hex(payload.dump())}};

  const std::string path = CacheFilePath(params, candidates);
  const std::string tmp = fmt::format("{}.tmp.{}", path, ::getpid());
  try {
    {
      std::ofstream ofs(tmp, std::ios::trunc);
      if (!ofs) {
        KLLM_LOG_WARNING << "HummingMoeSearchCache: cannot open tmp file " << tmp;
        return false;
      }
      ofs << doc.dump();
      if (!ofs) {
        KLLM_LOG_WARNING << "HummingMoeSearchCache: write failed " << tmp;
        return false;
      }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
      KLLM_LOG_WARNING << fmt::format("HummingMoeSearchCache: rename failed '{}': {}", tmp, ec.message());
      std::filesystem::remove(tmp, ec);
      return false;
    }
  } catch (const std::exception& e) {
    KLLM_LOG_WARNING << fmt::format("HummingMoeSearchCache: Save exception: {}", e.what());
    return false;
  }
  return true;
}

bool HummingMoeSearchCache::TryLoad(const HummingSearchCacheKeyParams& params, const KernelCandidates& candidates,
                                    HummingMoeSearchStatus::ProfiledSchedule& out_schedule) const {
  out_schedule = {};
  if (!Enabled() || Regenerate()) return false;
  const std::string path = CacheFilePath(params, candidates);
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) return false;  // common miss, no log

  try {
    nlohmann::json doc;
    {
      std::ifstream ifs(path);
      if (!ifs) return false;
      ifs >> doc;
    }
    if (doc.value("format_version", -1) != kSearchCacheFormatVersion) {
      KLLM_LOG_WARNING << "HummingMoeSearchCache: format_version mismatch, miss: " << path;
      return false;
    }
    if (doc.at("cache_key").get<std::string>() != ComputeCacheKey(params, candidates)) {
      KLLM_LOG_WARNING << "HummingMoeSearchCache: cache_key mismatch, miss: " << path;
      return false;
    }
    const nlohmann::json payload = nlohmann::json{{"params", doc.at("params")}, {"schedule", doc.at("schedule")}};
    if (doc.at("payload_sha256").get<std::string>() != Sha256Hex(payload.dump())) {
      KLLM_LOG_WARNING << "HummingMoeSearchCache: payload_sha256 mismatch, miss: " << path;
      return false;
    }

    // Resolve kernel_name → const KernelMeta* among current candidates.
    std::unordered_map<std::string, const llm_kernels::nvidia::humming::KernelMeta*> by_name;
    for (const auto* meta : candidates) {
      by_name[meta->kernel_name] = meta;
    }

    const auto& schedule_json = doc.at("schedule");
    HummingMoeSearchStatus::ProfiledSchedule sched;
    for (const auto& entry : schedule_json) {
      const int64_t size_m = entry.at("size_m").get<int64_t>();
      const std::string name = entry.at("kernel_name").get<std::string>();
      auto it = by_name.find(name);
      if (it == by_name.end()) {
        KLLM_LOG_WARNING << "HummingMoeSearchCache: kernel '" << name << "' not in current table, miss: " << path;
        return false;
      }
      if (!sched.size_ms.empty() && size_m <= sched.size_ms.back()) {
        KLLM_LOG_WARNING << "HummingMoeSearchCache: size_ms not strictly ascending, miss: " << path;
        return false;
      }
      sched.size_ms.push_back(size_m);
      sched.metas.push_back(it->second);
      sched.block_ms.push_back(it->second->block_m);
    }
    out_schedule = std::move(sched);
    return true;
  } catch (const std::exception& e) {
    KLLM_LOG_WARNING << fmt::format("HummingMoeSearchCache: load failed '{}': {}", path, e.what());
    return false;
  }
}

}  // namespace ksana_llm
