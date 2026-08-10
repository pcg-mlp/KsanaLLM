/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "batch_scheduler/global_cache_fetcher.h"
#include "batch_scheduler/global_cache_put_tracker.h"
#include "cache_manager/cache_manager_interface.h"
#include "ipc/functions/global_cache_put_done.h"
#include "ipc/functions/global_cache_query_done.h"
#include "loguru.hpp"

namespace ksana_llm {
namespace {

class MaxLogVerbosityEnv : public ::testing::Environment {
 public:
  void SetUp() override { loguru::g_stderr_verbosity = loguru::Verbosity_MAX; }
};
static auto* unused_log_env = ::testing::AddGlobalTestEnvironment(new MaxLogVerbosityEnv());

// ==================== GlobalCacheQueryDoneCallable 测试 ====================

class QueryDoneCallableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 同步模式的 fetcher（无 IPC），2 设备
    fetcher_ = std::make_shared<GlobalCacheFetcher>(2, nullptr);
    callable_ = std::make_unique<GlobalCacheQueryDoneCallable>();
    callable_->SetGlobalCacheFetcher(fetcher_);
  }

  std::shared_ptr<GlobalCacheFetcher> fetcher_;
  std::unique_ptr<GlobalCacheQueryDoneCallable> callable_;
};

TEST_F(QueryDoneCallableTest, CallWithFetcherForwardsResult) {
  // 先在 fetcher 中注册一个 pending 请求
  auto input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, req_ctx);
  auto infer_req = std::make_shared<InferRequest>(request, 0);
  infer_req->req_id = 42;
  infer_req->infer_stage = InferStage::kContext;
  infer_req->block_token_num = 16;
  for (int i = 0; i < 10; ++i) infer_req->output_tokens.push_back(i);
  infer_req->ResetPrefillingTokens();

  GlobalCacheFetcher::PendingRequest pending;
  pending.req = infer_req;
  pending.total_devices = 2;
  fetcher_->pending_requests_[42] = std::move(pending);

  // 通过 callable 模拟设备 0 完成
  GlobalCacheQueryDoneRequest req;
  req.req_id = 42;
  req.device_id = 0;
  req.matched_ranges = {0, 4};
  req.success = true;
  req.has_cached_tokens = false;

  GlobalCacheQueryDoneResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());

  // pending 中 devices_completed 应增加
  auto it = fetcher_->pending_requests_.find(42);
  ASSERT_NE(it, fetcher_->pending_requests_.end());
  EXPECT_EQ(it->second.devices_completed, 1);
}

TEST_F(QueryDoneCallableTest, CallWithoutFetcherNoThrow) {
  auto callable = std::make_unique<GlobalCacheQueryDoneCallable>();
  // 未设置 fetcher，不应崩溃

  GlobalCacheQueryDoneRequest req;
  req.req_id = 99;
  req.device_id = 0;
  req.success = false;

  GlobalCacheQueryDoneResponse resp;
  EXPECT_TRUE(callable->Call(&req, &resp, nullptr).OK());
}

TEST_F(QueryDoneCallableTest, SetGlobalCacheFetcher) {
  auto callable = std::make_unique<GlobalCacheQueryDoneCallable>();
  callable->SetGlobalCacheFetcher(fetcher_);
  EXPECT_EQ(callable->fetcher_, fetcher_);
}

// ==================== GlobalCachePutDoneCallable 测试 ====================

class MockCacheManagerForDone : public CacheManagerInterface {
 public:
  MOCK_METHOD(void, InitializeCachedBlocks, (), (override));
  MOCK_METHOD(std::shared_ptr<BlockAllocatorGroupInterface>, GetBlockAllocatorGroup, (), (const, override));
  MOCK_METHOD(size_t, GetUsableBlockNumber, (), (override));
  MOCK_METHOD(size_t, GetHostFreeBlockNumber, (), (override));
  MOCK_METHOD(size_t, GetUsedBlockNumber, (), (override));
  MOCK_METHOD(size_t, GetRequestStepBlockNumber, (int64_t, size_t, size_t), (override));
  MOCK_METHOD(size_t, GetRequestStepBlockNumberForOneNextToken, (int64_t), (override));
  MOCK_METHOD(size_t, GetRequestUsableBlockNumber, (int64_t), (override));
  MOCK_METHOD(Status, GetRequestPrefixBlockNumber,
              (int64_t, const std::vector<int>&, size_t, size_t&, size_t&, size_t&), (override));
  MOCK_METHOD(Status, GetRequestPrefixBlockNumberBeforeRemoteCacheLoading,
              (int64_t, const std::vector<int>&, size_t, size_t&, size_t&, size_t&, size_t&, size_t&, size_t&),
              (override));
  MOCK_METHOD(Status, AllocateBlocksForRemoteCache,
              (int64_t, std::vector<int>&, size_t, size_t, std::vector<int>&, std::vector<int>&), (override));
  MOCK_METHOD(Status, ConfirmRemoteCacheLoadingResult,
              (int64_t, const std::vector<int>&, size_t, const std::vector<size_t>&), (override));
  MOCK_METHOD(Status, AllocateRequestBlocks,
              (int64_t, size_t, std::vector<int>&, std::vector<int>&, std::vector<size_t>&, int&), (override));
  MOCK_METHOD(void, UpdateFlexibleCache,
              (int64_t, const std::vector<int>&, int, std::vector<FlexibleCachedCopyTask>&, size_t&), (override));
  MOCK_METHOD(Status, UpdateRequestTokens,
              (int64_t, const std::vector<int>&, size_t, std::vector<int>&, bool&, std::vector<int>&,
               std::vector<size_t>&),
              (override));
  MOCK_METHOD(void, DestroyFinishedRequest, (int64_t), (override));
  MOCK_METHOD(bool, IsPrefixCachingEnabled, (), (override));
};

class PutDoneCallableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_cm_ = std::make_shared<testing::NiceMock<MockCacheManagerForDone>>();
    tracker_ = std::make_shared<GlobalCachePutTracker>(2, mock_cm_);
    callable_ = std::make_unique<GlobalCachePutDoneCallable>();
    callable_->SetGlobalCachePutTracker(tracker_);
  }

  std::shared_ptr<testing::NiceMock<MockCacheManagerForDone>> mock_cm_;
  std::shared_ptr<GlobalCachePutTracker> tracker_;
  std::unique_ptr<GlobalCachePutDoneCallable> callable_;
};

TEST_F(PutDoneCallableTest, CallForwardsToTracker) {
  // 注册一个 prefill 请求到 tracker
  auto input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, req_ctx);
  auto infer_req = std::make_shared<InferRequest>(request, 0);
  infer_req->req_id = 55;
  infer_req->infer_stage = InferStage::kContext;
  tracker_->OnBatchDispatched({infer_req});

  EXPECT_TRUE(tracker_->DeferDestructionIfNeeded(55));

  // 通过 callable 报告设备 0 完成
  GlobalCachePutDoneRequest req;
  req.device_id = 0;
  req.req_id = 55;

  GlobalCachePutDoneResponse resp;
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());

  // 只有 1/2 设备完成，仍有 deferred
  EXPECT_TRUE(tracker_->HasDeferredRequests());

  // 报告设备 1 完成
  req.device_id = 1;
  EXPECT_CALL(*mock_cm_, DestroyFinishedRequest(55)).Times(1);
  EXPECT_TRUE(callable_->Call(&req, &resp, nullptr).OK());
  EXPECT_FALSE(tracker_->HasDeferredRequests());
}

TEST_F(PutDoneCallableTest, CallWithoutTrackerNoThrow) {
  auto callable = std::make_unique<GlobalCachePutDoneCallable>();

  GlobalCachePutDoneRequest req;
  req.device_id = 0;
  req.req_id = 77;

  GlobalCachePutDoneResponse resp;
  EXPECT_TRUE(callable->Call(&req, &resp, nullptr).OK());
}

TEST_F(PutDoneCallableTest, SetGlobalCachePutTracker) {
  auto callable = std::make_unique<GlobalCachePutDoneCallable>();
  callable->SetGlobalCachePutTracker(tracker_);
  EXPECT_EQ(callable->tracker_, tracker_);
}

// ==================== 序列化 roundtrip 测试 ====================

TEST(QueryDoneSerializationTest, RoundTrip) {
  GlobalCacheQueryDoneRequest original;
  original.req_id = 99;
  original.device_id = 3;
  original.matched_ranges = {0, 9, 15, 19};
  original.success = true;
  original.cached_tokens = {42, 43, 44};
  original.has_cached_tokens = true;

  std::vector<uint8_t> buffer(4096);
  size_t bytes = 0;
  ASSERT_TRUE(original.Serialize(buffer.data(), &bytes).OK());
  EXPECT_GT(bytes, 0u);

  GlobalCacheQueryDoneRequest restored;
  ASSERT_TRUE(restored.Deserialize(buffer.data()).OK());
  EXPECT_EQ(restored.req_id, 99);
  EXPECT_EQ(restored.device_id, 3);
  EXPECT_EQ(restored.matched_ranges, original.matched_ranges);
  EXPECT_TRUE(restored.success);
  EXPECT_EQ(restored.cached_tokens, original.cached_tokens);
  EXPECT_TRUE(restored.has_cached_tokens);
}

TEST(QueryDoneSerializationTest, EmptyFields) {
  GlobalCacheQueryDoneRequest original;
  original.req_id = -1;
  original.device_id = 0;
  original.success = false;
  original.has_cached_tokens = false;

  std::vector<uint8_t> buffer(4096);
  size_t bytes = 0;
  ASSERT_TRUE(original.Serialize(buffer.data(), &bytes).OK());

  GlobalCacheQueryDoneRequest restored;
  ASSERT_TRUE(restored.Deserialize(buffer.data()).OK());
  EXPECT_EQ(restored.req_id, -1);
  EXPECT_TRUE(restored.matched_ranges.empty());
  EXPECT_FALSE(restored.success);
  EXPECT_TRUE(restored.cached_tokens.empty());
  EXPECT_FALSE(restored.has_cached_tokens);
}

TEST(PutDoneSerializationTest, RoundTrip) {
  GlobalCachePutDoneRequest original;
  original.device_id = 7;
  original.req_id = 123;

  std::vector<uint8_t> buffer(256);
  size_t bytes = 0;
  ASSERT_TRUE(original.Serialize(buffer.data(), &bytes).OK());
  EXPECT_GT(bytes, 0u);

  GlobalCachePutDoneRequest restored;
  ASSERT_TRUE(restored.Deserialize(buffer.data()).OK());
  EXPECT_EQ(restored.device_id, 7);
  EXPECT_EQ(restored.req_id, 123);
}

TEST(PutDoneSerializationTest, DefaultValues) {
  GlobalCachePutDoneRequest original;

  std::vector<uint8_t> buffer(256);
  size_t bytes = 0;
  ASSERT_TRUE(original.Serialize(buffer.data(), &bytes).OK());

  GlobalCachePutDoneRequest restored;
  ASSERT_TRUE(restored.Deserialize(buffer.data()).OK());
  EXPECT_EQ(restored.device_id, -1);
  EXPECT_EQ(restored.req_id, -1);
}

}  // namespace
}  // namespace ksana_llm
