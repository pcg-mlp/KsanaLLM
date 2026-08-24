/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "ipc/functions/global_cache_put_done.h"
#include "ipc/functions/global_cache_query.h"
#include "ipc/functions/global_cache_query_done.h"
#include "loguru.hpp"

namespace ksana_llm {

namespace {
class MaxLogVerbosityEnv : public ::testing::Environment {
 public:
  void SetUp() override { loguru::g_stderr_verbosity = loguru::Verbosity_MAX; }
};
static auto* unused_log_env = ::testing::AddGlobalTestEnvironment(new MaxLogVerbosityEnv());
}  // namespace

class GlobalCacheIpcSerializationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    buf_size_ = 64 * 1024;
    buf_ = new uint8_t[buf_size_];
    std::memset(buf_, 0, buf_size_);
  }
  void TearDown() override { delete[] buf_; }

  size_t buf_size_;
  uint8_t* buf_;
};

// ---------- GlobalCacheQueryRequest ----------

TEST_F(GlobalCacheIpcSerializationTest, QueryRequestRoundTrip) {
  GlobalCacheQueryRequest req;
  req.req_id = 12345;
  req.token_ids = {10, 20, 30, 40, 50};
  req.local_cached_len = 8;
  req.block_size = 4;
  req.kv_cache_block_ids = {100, 101, 102};
  req.enqueue_wall_us = 1710000000123;

  size_t bytes = 0;
  ASSERT_TRUE(req.Serialize(buf_, &bytes).OK());
  ASSERT_GT(bytes, 0u);

  GlobalCacheQueryRequest restored;
  ASSERT_TRUE(restored.Deserialize(buf_).OK());

  EXPECT_EQ(restored.req_id, req.req_id);
  EXPECT_EQ(restored.token_ids, req.token_ids);
  EXPECT_EQ(restored.local_cached_len, req.local_cached_len);
  EXPECT_EQ(restored.block_size, req.block_size);
  EXPECT_EQ(restored.kv_cache_block_ids, req.kv_cache_block_ids);
  EXPECT_EQ(restored.enqueue_wall_us, req.enqueue_wall_us);
}

TEST_F(GlobalCacheIpcSerializationTest, QueryRequestEmptyVectors) {
  GlobalCacheQueryRequest req;
  req.req_id = 0;
  req.token_ids = {};
  req.local_cached_len = 0;
  req.block_size = 0;
  req.kv_cache_block_ids = {};

  size_t bytes = 0;
  ASSERT_TRUE(req.Serialize(buf_, &bytes).OK());

  GlobalCacheQueryRequest restored;
  ASSERT_TRUE(restored.Deserialize(buf_).OK());

  EXPECT_EQ(restored.req_id, 0);
  EXPECT_TRUE(restored.token_ids.empty());
  EXPECT_EQ(restored.local_cached_len, 0u);
  EXPECT_EQ(restored.block_size, 0);
  EXPECT_TRUE(restored.kv_cache_block_ids.empty());
}

TEST_F(GlobalCacheIpcSerializationTest, QueryRequestLargeTokenIds) {
  GlobalCacheQueryRequest req;
  req.req_id = 99999;
  req.token_ids.resize(2048);
  for (int i = 0; i < 2048; ++i) req.token_ids[i] = i;
  req.local_cached_len = 512;
  req.block_size = 16;
  req.kv_cache_block_ids.resize(128);
  for (int i = 0; i < 128; ++i) req.kv_cache_block_ids[i] = 1000 + i;

  size_t bytes = 0;
  ASSERT_TRUE(req.Serialize(buf_, &bytes).OK());

  GlobalCacheQueryRequest restored;
  ASSERT_TRUE(restored.Deserialize(buf_).OK());

  EXPECT_EQ(restored.req_id, req.req_id);
  EXPECT_EQ(restored.token_ids, req.token_ids);
  EXPECT_EQ(restored.local_cached_len, req.local_cached_len);
  EXPECT_EQ(restored.block_size, req.block_size);
  EXPECT_EQ(restored.kv_cache_block_ids, req.kv_cache_block_ids);
}

// ---------- GlobalCachePutDoneRequest ----------

TEST_F(GlobalCacheIpcSerializationTest, PutDoneRequestRoundTrip) {
  GlobalCachePutDoneRequest req;
  req.device_id = 3;
  req.req_id = 67890;
  req.fa_put_blocks = 12;
  req.swa_put_blocks = 4;

  size_t bytes = 0;
  ASSERT_TRUE(req.Serialize(buf_, &bytes).OK());
  ASSERT_GT(bytes, 0u);

  GlobalCachePutDoneRequest restored;
  ASSERT_TRUE(restored.Deserialize(buf_).OK());

  EXPECT_EQ(restored.device_id, req.device_id);
  EXPECT_EQ(restored.req_id, req.req_id);
  EXPECT_EQ(restored.fa_put_blocks, req.fa_put_blocks);
  EXPECT_EQ(restored.swa_put_blocks, req.swa_put_blocks);
}

TEST_F(GlobalCacheIpcSerializationTest, PutDoneRequestZeroValues) {
  GlobalCachePutDoneRequest req;
  req.device_id = 0;
  req.req_id = 0;

  size_t bytes = 0;
  ASSERT_TRUE(req.Serialize(buf_, &bytes).OK());

  GlobalCachePutDoneRequest restored;
  ASSERT_TRUE(restored.Deserialize(buf_).OK());

  EXPECT_EQ(restored.device_id, 0);
  EXPECT_EQ(restored.req_id, 0);
}

TEST_F(GlobalCacheIpcSerializationTest, PutDoneRequestNegativeValues) {
  GlobalCachePutDoneRequest req;
  req.device_id = -1;
  req.req_id = -1;

  size_t bytes = 0;
  ASSERT_TRUE(req.Serialize(buf_, &bytes).OK());

  GlobalCachePutDoneRequest restored;
  ASSERT_TRUE(restored.Deserialize(buf_).OK());

  EXPECT_EQ(restored.device_id, -1);
  EXPECT_EQ(restored.req_id, -1);
}

// ---------- GlobalCacheQueryDoneRequest ----------

TEST_F(GlobalCacheIpcSerializationTest, QueryDoneRequestRoundTrip) {
  GlobalCacheQueryDoneRequest req;
  req.req_id = 54321;
  req.device_id = 2;
  req.matched_ranges = {0, 127, 256, 383};
  req.success = true;
  req.cached_tokens = {42, 99, 7};
  req.has_cached_tokens = true;
  req.wait_us = 12000;
  req.qf_us = 34000;
  req.token_us = 5000;
  req.report_us = 1000;
  req.call_us = 40000;
  req.report_wall_us = 1710000000456;
  req.swa_aligned_hit_len = 128;
  req.swa_filled_block_idxs = {1, 2, 3};

  size_t bytes = 0;
  ASSERT_TRUE(req.Serialize(buf_, &bytes).OK());
  ASSERT_GT(bytes, 0u);

  GlobalCacheQueryDoneRequest restored;
  ASSERT_TRUE(restored.Deserialize(buf_).OK());

  EXPECT_EQ(restored.req_id, req.req_id);
  EXPECT_EQ(restored.device_id, req.device_id);
  EXPECT_EQ(restored.matched_ranges, req.matched_ranges);
  EXPECT_EQ(restored.success, req.success);
  EXPECT_EQ(restored.cached_tokens, req.cached_tokens);
  EXPECT_EQ(restored.has_cached_tokens, req.has_cached_tokens);
  EXPECT_EQ(restored.wait_us, req.wait_us);
  EXPECT_EQ(restored.qf_us, req.qf_us);
  EXPECT_EQ(restored.token_us, req.token_us);
  EXPECT_EQ(restored.report_us, req.report_us);
  EXPECT_EQ(restored.call_us, req.call_us);
  EXPECT_EQ(restored.report_wall_us, req.report_wall_us);
  EXPECT_EQ(restored.swa_aligned_hit_len, req.swa_aligned_hit_len);
  EXPECT_EQ(restored.swa_filled_block_idxs, req.swa_filled_block_idxs);
}

TEST_F(GlobalCacheIpcSerializationTest, QueryDoneRequestEmptyResult) {
  GlobalCacheQueryDoneRequest req;
  req.req_id = 111;
  req.device_id = 0;
  req.matched_ranges = {};
  req.success = false;
  req.cached_tokens = {};
  req.has_cached_tokens = false;

  size_t bytes = 0;
  ASSERT_TRUE(req.Serialize(buf_, &bytes).OK());

  GlobalCacheQueryDoneRequest restored;
  ASSERT_TRUE(restored.Deserialize(buf_).OK());

  EXPECT_EQ(restored.req_id, req.req_id);
  EXPECT_EQ(restored.device_id, req.device_id);
  EXPECT_TRUE(restored.matched_ranges.empty());
  EXPECT_FALSE(restored.success);
  EXPECT_TRUE(restored.cached_tokens.empty());
  EXPECT_FALSE(restored.has_cached_tokens);
}

TEST_F(GlobalCacheIpcSerializationTest, QueryDoneRequestLargeMatchedRanges) {
  GlobalCacheQueryDoneRequest req;
  req.req_id = 222;
  req.device_id = 7;
  req.success = true;
  req.has_cached_tokens = false;
  for (int i = 0; i < 100; ++i) {
    req.matched_ranges.push_back(i * 128);
    req.matched_ranges.push_back(i * 128 + 127);
  }

  size_t bytes = 0;
  ASSERT_TRUE(req.Serialize(buf_, &bytes).OK());

  GlobalCacheQueryDoneRequest restored;
  ASSERT_TRUE(restored.Deserialize(buf_).OK());

  EXPECT_EQ(restored.matched_ranges, req.matched_ranges);
  EXPECT_EQ(restored.success, true);
  EXPECT_EQ(restored.has_cached_tokens, false);
}

}  // namespace ksana_llm
