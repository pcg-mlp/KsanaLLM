/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "csrc/kernels/nvidia/others/humming/launcher/kernel_meta.h"
#include "layers/nvidia/humming_search_cache.h"
#include "nlohmann/json.hpp"

#ifdef ENABLE_CUDA

#  include <cuda_runtime.h>

namespace ksana_llm {

using llm_kernels::nvidia::humming::KernelMeta;

class HummingSearchCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major * 10 + prop.minor != 90) {
      GTEST_SKIP() << "Humming kernels are built for sm90a only, skipping on sm_" << prop.major * 10 + prop.minor;
    }
    dir_ = "/tmp/opencode/humming_search_cache_test_" + std::to_string(::getpid());
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
    cache_ = std::make_unique<HummingMoeSearchCache>(dir_);
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  static KernelMeta MakeMeta(const std::string& name, int64_t n, int64_t k, int dt = 1, int gt = 1, int block_m = 64) {
    KernelMeta m{};
    m.kernel_name = name;
    m.size_n = n;
    m.size_k = k;
    m.compute_dtype = static_cast<llm_kernels::nvidia::humming::ComputeDtype>(dt);
    m.gemm_type = static_cast<llm_kernels::nvidia::humming::GemmType>(gt);
    m.block_m = block_m;
    return m;
  }

  static HummingSearchCacheKeyParams MakeParams() {
    HummingSearchCacheKeyParams p;
    p.n = 3072;
    p.k = 7168;
    p.e = 128;
    p.dt = 1;
    p.gt = 1;
    p.top_k = 8;
    p.size_m_multiplier = 1;
    p.ep_size = 8;
    p.using_deepep = false;
    p.rank = 0;
    p.profile_iters = 15;
    p.warmup_iters = 5;
    p.m_thresh1 = 512;
    p.gpu_name = "TEST-GPU";
    p.sm_count = 78;
    p.cc = "9.0";
    return p;
  }

  static std::vector<KernelMeta> MakeTable() {
    return {MakeMeta("_ZN7humming8x128x64a", 3072, 7168, 1, 1, 64),
            MakeMeta("_ZN7humming16x128x64b", 3072, 7168, 1, 1, 128),
            MakeMeta("_ZN7humming8x128x64_w4a16", 3072, 7168, 0, 1, 64)};  // BF16, must not affect dt=1 key
  }

  // Pre-filtered candidates for MakeParams() (what GetHummingKernelCandidates
  // would return for n=3072 k=7168 dt=1 gt=1) — the cache consumes these as-is.
  static HummingMoeSearchCache::KernelCandidates MakeCandidates(const std::vector<KernelMeta>& table) {
    return {&table[0], &table[1]};
  }

  // Candidates built from an arbitrary table (filters like the real accessor).
  static HummingMoeSearchCache::KernelCandidates FilterCandidates(const std::vector<KernelMeta>& table) {
    HummingMoeSearchCache::KernelCandidates out;
    for (const auto& m : table) {
      if (m.size_n != 3072 || m.size_k != 7168 || static_cast<int>(m.compute_dtype) != 1 ||
          static_cast<int>(m.gemm_type) != 1) {
        continue;
      }
      out.push_back(&m);
    }
    return out;
  }

  std::string dir_;
  std::unique_ptr<HummingMoeSearchCache> cache_;
};

class EnvGuard {
 public:
  explicit EnvGuard(const char* name) : name_(name) {
    const char* old = std::getenv(name);
    existed_ = (old != nullptr);
    old_value_ = old ? old : "";
  }
  ~EnvGuard() {
    if (existed_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string old_value_;
  bool existed_;
};

HummingMoeSearchStatus::ProfiledSchedule MakeSchedule(const std::vector<KernelMeta>& table) {
  HummingMoeSearchStatus::ProfiledSchedule s;
  s.size_ms = {1, 2};
  s.metas = {&table[0], &table[1]};
  s.block_ms = {table[0].block_m, table[1].block_m};
  return s;
}

TEST_F(HummingSearchCacheTest, KeyDeterministicAndSensitive) {
  const auto table = MakeTable();
  const auto cands = MakeCandidates(table);
  const std::string key = cache_->ComputeCacheKey(MakeParams(), cands);
  EXPECT_EQ(key.size(), 64u);
  EXPECT_EQ(key, cache_->ComputeCacheKey(MakeParams(), cands));

  // Every key-identity field change must change the key.
  std::vector<std::function<HummingSearchCacheKeyParams()>> mutations = {
      [] {
        auto p = MakeParams();
        p.n = 4096;
        return p;
      },
      [] {
        auto p = MakeParams();
        p.k = 4096;
        return p;
      },
      [] {
        auto p = MakeParams();
        p.e = 256;
        return p;
      },
      [] {
        auto p = MakeParams();
        p.dt = 0;
        return p;
      },
      [] {
        auto p = MakeParams();
        p.gt = 2;
        return p;
      },
      [] {
        auto p = MakeParams();
        p.top_k = 6;
        return p;
      },
      [] {
        auto p = MakeParams();
        p.size_m_multiplier = 6;
        return p;
      },
      [] {
        auto p = MakeParams();
        p.ep_size = 1;
        return p;
      },
      [] {
        auto p = MakeParams();
        p.using_deepep = true;
        return p;
      },
      [] {
        auto p = MakeParams();
        p.gpu_name = "NVIDIA H200";
        return p;
      },
      [] {
        auto p = MakeParams();
        p.sm_count = 132;
        return p;
      },
  };
  for (const auto& mut : mutations) {
    EXPECT_NE(key, cache_->ComputeCacheKey(mut(), cands));
  }

  // Diagnostics fields must NOT change the key.
  {
    auto p = MakeParams();
    p.rank = 3;
    p.profile_iters = 30;
    p.cc = "10.0";
    EXPECT_EQ(key, cache_->ComputeCacheKey(p, cands));
  }

  // Candidate-set changes must change the key (FP4_COMPILE_EXTEND scenario).
  {
    auto extended = MakeTable();
    extended.push_back(MakeMeta("_ZN7humming32x128x64_ext", 3072, 7168, 1, 1, 32));
    EXPECT_NE(key, cache_->ComputeCacheKey(MakeParams(), FilterCandidates(extended)));
  }
  // Removing a candidate also changes the key.
  {
    std::vector<KernelMeta> reduced = {MakeTable()[0]};
    EXPECT_NE(key, cache_->ComputeCacheKey(MakeParams(), FilterCandidates(reduced)));
  }
  // Adding a W4A16 (dt=0) kernel must NOT change the dt=1 key.
  {
    auto with_a16 = MakeTable();
    with_a16.push_back(MakeMeta("_ZN7humming8x128x64_w4a16_extra", 3072, 7168, 0, 1, 64));
    EXPECT_EQ(key, cache_->ComputeCacheKey(MakeParams(), FilterCandidates(with_a16)));
  }
  // Table order must not matter (names sorted before hashing).
  {
    auto shuffled = MakeTable();
    std::reverse(shuffled.begin(), shuffled.end());
    EXPECT_EQ(key, cache_->ComputeCacheKey(MakeParams(), FilterCandidates(shuffled)));
  }
}

TEST_F(HummingSearchCacheTest, Sha256HexKnownVector) {
  // sha256("abc")
  EXPECT_EQ(HummingMoeSearchCache::Sha256Hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST_F(HummingSearchCacheTest, CacheFilePathFormat) {
  const auto table = MakeTable();
  const auto cands = MakeCandidates(table);
  const std::string path = cache_->CacheFilePath(MakeParams(), cands);
  const std::string key = cache_->ComputeCacheKey(MakeParams(), cands);
  EXPECT_EQ(path, dir_ + "/humming_search_" + key.substr(0, 16) + "_n3072_k7168_rank0.json");
}

TEST_F(HummingSearchCacheTest, SaveLoadRoundTrip) {
  const auto table = MakeTable();
  const auto cands = MakeCandidates(table);
  auto sched = MakeSchedule(table);
  ASSERT_TRUE(cache_->Save(MakeParams(), cands, sched));

  HummingMoeSearchStatus::ProfiledSchedule loaded;
  ASSERT_TRUE(cache_->TryLoad(MakeParams(), cands, loaded));
  ASSERT_EQ(loaded.size_ms, sched.size_ms);
  ASSERT_EQ(loaded.block_ms, sched.block_ms);
  ASSERT_EQ(loaded.metas.size(), 2u);
  EXPECT_EQ(loaded.metas[0]->kernel_name, table[0].kernel_name);
  EXPECT_EQ(loaded.metas[1]->kernel_name, table[1].kernel_name);
}

TEST_F(HummingSearchCacheTest, TryLoadMissWhenNoFile) {
  const auto table = MakeTable();  // must outlive cands (pointers into it)
  const auto cands = MakeCandidates(table);
  HummingMoeSearchStatus::ProfiledSchedule loaded;
  EXPECT_FALSE(cache_->TryLoad(MakeParams(), cands, loaded));
}

TEST_F(HummingSearchCacheTest, CorruptFileRejected) {
  const auto table = MakeTable();
  const auto cands = MakeCandidates(table);
  ASSERT_TRUE(cache_->Save(MakeParams(), cands, MakeSchedule(table)));
  const std::string path = cache_->CacheFilePath(MakeParams(), cands);
  HummingMoeSearchStatus::ProfiledSchedule loaded;

  // a) truncated / garbage json
  { std::ofstream(path) << "not json at all"; }
  EXPECT_FALSE(cache_->TryLoad(MakeParams(), cands, loaded));

  // b) tampered payload with stale payload_sha256
  ASSERT_TRUE(cache_->Save(MakeParams(), cands, MakeSchedule(table)));
  {
    nlohmann::json doc;
    std::ifstream(path) >> doc;
    doc["schedule"][0]["size_m"] = 999;
    std::ofstream(path) << doc.dump();
  }
  EXPECT_FALSE(cache_->TryLoad(MakeParams(), cands, loaded));

  // c) tampered cache_key (payload hash untouched → still consistent)
  ASSERT_TRUE(cache_->Save(MakeParams(), cands, MakeSchedule(table)));
  {
    nlohmann::json doc;
    std::ifstream(path) >> doc;
    doc["cache_key"] = std::string(64, '0');
    std::ofstream(path) << doc.dump();
  }
  EXPECT_FALSE(cache_->TryLoad(MakeParams(), cands, loaded));

  // d) bumped format_version
  ASSERT_TRUE(cache_->Save(MakeParams(), cands, MakeSchedule(table)));
  {
    nlohmann::json doc;
    std::ifstream(path) >> doc;
    doc["format_version"] = HummingMoeSearchCache::kSearchCacheFormatVersion + 1;
    std::ofstream(path) << doc.dump();
  }
  EXPECT_FALSE(cache_->TryLoad(MakeParams(), cands, loaded));

  // e) kernel missing from current candidates (e.g. heuristic changed)
  ASSERT_TRUE(cache_->Save(MakeParams(), cands, MakeSchedule(table)));
  {
    // schedule references table[1]; reduced candidate set lacks it
    HummingMoeSearchCache::KernelCandidates reduced = {&table[0]};
    EXPECT_FALSE(cache_->TryLoad(MakeParams(), reduced, loaded));
  }

  // f) self-consistent but non-ascending size_ms (payload hash recomputed)
  ASSERT_TRUE(cache_->Save(MakeParams(), cands, MakeSchedule(table)));
  {
    nlohmann::json doc;
    std::ifstream(path) >> doc;
    std::swap(doc["schedule"][0], doc["schedule"][1]);
    doc["payload_sha256"] = HummingMoeSearchCache::Sha256Hex(
        nlohmann::json({{"params", doc["params"]}, {"schedule", doc["schedule"]}}).dump());
    std::ofstream(path) << doc.dump();
  }
  EXPECT_FALSE(cache_->TryLoad(MakeParams(), cands, loaded));
}

TEST_F(HummingSearchCacheTest, TryLoadNoThrowOnBadDir) {
  // Parent path component is a file → filesystem::exists would throw ENOTDIR
  // if the throwing overload were used. Must degrade to a plain miss.
  const auto table = MakeTable();  // must outlive cands (pointers into it)
  const auto cands = MakeCandidates(table);
  { std::ofstream(dir_ + "/notadir") << "i am a file"; }
  HummingMoeSearchCache cache2(dir_ + "/notadir/sub");
  HummingMoeSearchStatus::ProfiledSchedule loaded;
  ASSERT_NO_THROW(EXPECT_FALSE(cache2.TryLoad(MakeParams(), cands, loaded)));
}

TEST_F(HummingSearchCacheTest, OverwriteSelfHeal) {
  const auto table = MakeTable();
  const auto cands = MakeCandidates(table);
  HummingMoeSearchStatus::ProfiledSchedule loaded;
  ASSERT_TRUE(cache_->Save(MakeParams(), cands, MakeSchedule(table)));
  { std::ofstream(cache_->CacheFilePath(MakeParams(), cands)) << "corrupted"; }
  EXPECT_FALSE(cache_->TryLoad(MakeParams(), cands, loaded));
  ASSERT_TRUE(cache_->Save(MakeParams(), cands, MakeSchedule(table)));  // overwrite heals
  EXPECT_TRUE(cache_->TryLoad(MakeParams(), cands, loaded));
}

// The default-constructed cache must derive its directory from the shared
// JIT-side resolver (single source of truth), not from its own env parsing.
TEST_F(HummingSearchCacheTest, DefaultDirDerivesFromSharedResolver) {
  EnvGuard g("KSANA_HUMMING_KERNEL_PATH");
  setenv("KSANA_HUMMING_KERNEL_PATH", dir_.c_str(), 1);
  HummingMoeSearchCache default_cache;
  const auto table = MakeTable();  // must outlive cands (pointers into it)
  const auto cands = MakeCandidates(table);
  const std::string path = default_cache.CacheFilePath(MakeParams(), cands);
  EXPECT_EQ(path, dir_ + "/humming_search_cache/humming_search_" +
                      default_cache.ComputeCacheKey(MakeParams(), cands).substr(0, 16) + "_n3072_k7168_rank0.json");
}

TEST_F(HummingSearchCacheTest, EnvDisabledSkipsReadAndWrite) {
  EnvGuard g("HUMMING_SEARCH_CACHE");
  setenv("HUMMING_SEARCH_CACHE", "0", 1);
  const auto table = MakeTable();
  const auto cands = MakeCandidates(table);
  HummingMoeSearchStatus::ProfiledSchedule loaded;
  EXPECT_FALSE(cache_->Save(MakeParams(), cands, MakeSchedule(table)));
  EXPECT_TRUE(std::filesystem::exists(cache_->CacheFilePath(MakeParams(), cands)) == false);
  EXPECT_FALSE(cache_->TryLoad(MakeParams(), cands, loaded));
}

TEST_F(HummingSearchCacheTest, EnvRegenerateSkipsLoadButSaves) {
  const auto table = MakeTable();
  const auto cands = MakeCandidates(table);
  // Seed a valid file with regenerate unset.
  ASSERT_TRUE(cache_->Save(MakeParams(), cands, MakeSchedule(table)));

  EnvGuard g("HUMMING_SEARCH_REGENERATE");
  setenv("HUMMING_SEARCH_REGENERATE", "1", 1);
  HummingMoeSearchStatus::ProfiledSchedule loaded;
  EXPECT_FALSE(cache_->TryLoad(MakeParams(), cands, loaded));  // forced re-search

  // Re-search result differs (single bucket, different kernel) → Save overwrites.
  HummingMoeSearchStatus::ProfiledSchedule searched;
  searched.size_ms = {1};
  searched.metas = {&table[1]};
  searched.block_ms = {table[1].block_m};
  ASSERT_TRUE(cache_->Save(MakeParams(), cands, searched));
  unsetenv("HUMMING_SEARCH_REGENERATE");
  ASSERT_TRUE(cache_->TryLoad(MakeParams(), cands, loaded));  // loads the overwritten content
  EXPECT_EQ(loaded.size_ms, searched.size_ms);
  EXPECT_EQ(loaded.metas[0]->kernel_name, table[1].kernel_name);
}

TEST_F(HummingSearchCacheTest, AtomicWriteNoTmpResidue) {
  const auto table = MakeTable();
  const auto cands = MakeCandidates(table);
  ASSERT_TRUE(cache_->Save(MakeParams(), cands, MakeSchedule(table)));
  std::vector<std::string> entries;
  for (const auto& e : std::filesystem::directory_iterator(dir_)) entries.push_back(e.path().string());
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0], cache_->CacheFilePath(MakeParams(), cands));
}

}  // namespace ksana_llm

#endif  // ENABLE_CUDA
