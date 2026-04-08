/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <vector>

#include "ipc/functions/global_cache_query.h"
#include "loguru.hpp"

namespace ksana_llm {

namespace {
class MaxLogVerbosityEnv : public ::testing::Environment {
 public:
  void SetUp() override { loguru::g_stderr_verbosity = loguru::Verbosity_MAX; }
};
static auto* unused_log_env = ::testing::AddGlobalTestEnvironment(new MaxLogVerbosityEnv());
}  // namespace

// IsFullHit 是 GlobalCacheQueryCallable 的 private static 方法，
// 编译时使用 -fno-access-control 使其可被测试访问。
class IsFullHitTest : public ::testing::Test {
 protected:
  static bool IsFullHit(const std::vector<int>& matched_ranges, int total_tokens, int local_cached_len) {
    return GlobalCacheQueryCallable::IsFullHit(matched_ranges, total_tokens, local_cached_len);
  }
};

TEST_F(IsFullHitTest, EmptyRangesNotFullHit) {
  EXPECT_FALSE(IsFullHit({}, 100, 0));
}

TEST_F(IsFullHitTest, ZeroTotalTokensNotFullHit) {
  EXPECT_FALSE(IsFullHit({0, 99}, 0, 0));
}

TEST_F(IsFullHitTest, SingleRangeFullCoverage) {
  EXPECT_TRUE(IsFullHit({0, 99}, 100, 0));
}

TEST_F(IsFullHitTest, SingleRangePartialCoverage) {
  EXPECT_FALSE(IsFullHit({0, 49}, 100, 0));
}

TEST_F(IsFullHitTest, LocalCachedLenSkipsPrefix) {
  EXPECT_TRUE(IsFullHit({50, 99}, 100, 50));
}

TEST_F(IsFullHitTest, LocalCachedLenWithGap) {
  EXPECT_FALSE(IsFullHit({60, 99}, 100, 50));
}

TEST_F(IsFullHitTest, MultipleRangesFullCoverage) {
  EXPECT_TRUE(IsFullHit({0, 49, 50, 99}, 100, 0));
}

TEST_F(IsFullHitTest, MultipleRangesWithGap) {
  EXPECT_FALSE(IsFullHit({0, 29, 50, 99}, 100, 0));
}

TEST_F(IsFullHitTest, OverlappingRangesFullCoverage) {
  EXPECT_TRUE(IsFullHit({0, 59, 40, 99}, 100, 0));
}

TEST_F(IsFullHitTest, ExactlyOneToken) {
  EXPECT_TRUE(IsFullHit({0, 0}, 1, 0));
}

TEST_F(IsFullHitTest, LocalCachedEqualsTotalTokens) {
  EXPECT_FALSE(IsFullHit({}, 100, 100));
}

TEST_F(IsFullHitTest, RangeCoversMoreThanNeeded) {
  EXPECT_TRUE(IsFullHit({0, 199}, 100, 0));
}

TEST_F(IsFullHitTest, MultipleContiguousRangesWithLocalCache) {
  EXPECT_TRUE(IsFullHit({64, 127, 128, 255}, 256, 64));
}

}  // namespace ksana_llm
