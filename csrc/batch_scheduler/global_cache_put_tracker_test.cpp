/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "batch_scheduler/global_cache_put_tracker.h"
#include "cache_manager/cache_manager_interface.h"
#include "loguru.hpp"
#include "runtime/infer_request.h"
#include "utils/request.h"

namespace ksana_llm {
namespace {

class MaxLogVerbosityEnv : public ::testing::Environment {
 public:
  void SetUp() override { loguru::g_stderr_verbosity = loguru::Verbosity_MAX; }
};
static auto* unused_log_env = ::testing::AddGlobalTestEnvironment(new MaxLogVerbosityEnv());

class MockCacheManager : public CacheManagerInterface {
 public:
  MOCK_METHOD(void, InitializeCachedBlocks, (), (override));
  MOCK_METHOD(std::shared_ptr<BlockAllocatorGroupInterface>, GetBlockAllocatorGroup, (), (const, override));
  MOCK_METHOD(size_t, GetUsableBlockNumber, (), (override));
  MOCK_METHOD(size_t, GetFutureFreeBlockNumber, (), (override));
  MOCK_METHOD(size_t, GetHostFreeBlockNumber, (), (override));
  MOCK_METHOD(size_t, GetUsedBlockNumber, (), (override));
  MOCK_METHOD(size_t, GetRequestStepBlockNumber, (int64_t, size_t), (override));
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
              (int64_t, size_t, std::vector<int>&, std::vector<int>&, std::vector<size_t>&), (override));
  MOCK_METHOD(void, UpdateFlexibleCache,
              (int64_t, const std::vector<int>&, int, std::vector<FlexibleCachedCopyTask>&, size_t&), (override));
  MOCK_METHOD(Status, UpdateRequestTokens,
              (int64_t, const std::vector<int>&, size_t, std::vector<int>&, bool&, std::vector<int>&,
               std::vector<size_t>&),
              (override));
  MOCK_METHOD(Status, GetRequestFreeableBlockNum, (int64_t, size_t&), (override));
  MOCK_METHOD(Status, GetRequestNeededBlockNumForOneNextToken, (int64_t, size_t&), (override));
  MOCK_METHOD(void, SetSwapCallbackFunc, (AsyncSwapCallbackFunc), (override));
  MOCK_METHOD(Status, SwapoutRequestAsync,
              (int64_t, size_t&, size_t&, std::vector<int>&, std::vector<int>&), (override));
  MOCK_METHOD(Status, SwapinRequestAsync,
              (int64_t, size_t&, std::vector<int>&, std::vector<int>&, std::vector<int>&), (override));
  MOCK_METHOD(Status, WaitSwapoutRequests, (std::vector<int64_t>&, size_t&, bool), (override));
  MOCK_METHOD(Status, WaitSwapinRequests, (std::vector<int64_t>&, size_t&, bool), (override));
  MOCK_METHOD(Status, WaitSwapoutRequests,
              (const std::unordered_set<int64_t>&, std::vector<int64_t>&, size_t&, bool), (override));
  MOCK_METHOD(Status, WaitSwapinRequests,
              (const std::unordered_set<int64_t>&, std::vector<int64_t>&, size_t&, bool), (override));
  MOCK_METHOD(Status, MergeSwapoutRequest, (int64_t), (override));
  MOCK_METHOD(Status, MergeSwapinRequest, (int64_t, std::vector<int>&, std::vector<int>&, std::vector<size_t>&),
              (override));
  MOCK_METHOD(void, DestroySwappedRequest, (int64_t), (override));
  MOCK_METHOD(void, DestroyFinishedRequest, (int64_t), (override));
  MOCK_METHOD(bool, IsPrefixCachingEnabled, (), (override));
};

std::shared_ptr<InferRequest> MakeInferRequest(int64_t req_id, InferStage stage) {
  auto input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, req_ctx);
  auto infer_req = std::make_shared<InferRequest>(request, 0);
  infer_req->req_id = req_id;
  infer_req->infer_stage = stage;
  return infer_req;
}

class PutTrackerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_cm_ = std::make_shared<testing::NiceMock<MockCacheManager>>();
    tracker_ = std::make_unique<GlobalCachePutTracker>(kDeviceCount, mock_cm_);
  }

  static constexpr int kDeviceCount = 2;
  std::shared_ptr<testing::NiceMock<MockCacheManager>> mock_cm_;
  std::unique_ptr<GlobalCachePutTracker> tracker_;
};

TEST_F(PutTrackerTest, DecodeRequestNotTracked) {
  auto req = MakeInferRequest(100, InferStage::kDecode);
  tracker_->OnBatchDispatched({req});
  EXPECT_FALSE(tracker_->DeferDestructionIfNeeded(100));
}

TEST_F(PutTrackerTest, PrefillRequestTracked) {
  auto req = MakeInferRequest(200, InferStage::kContext);
  tracker_->OnBatchDispatched({req});
  EXPECT_TRUE(tracker_->DeferDestructionIfNeeded(200));
  EXPECT_TRUE(tracker_->HasDeferredRequests());
}

TEST_F(PutTrackerTest, AllDevicesDoneTriggersDestroyFinishedRequest) {
  auto req = MakeInferRequest(300, InferStage::kContext);
  tracker_->OnBatchDispatched({req});
  EXPECT_TRUE(tracker_->DeferDestructionIfNeeded(300));
  EXPECT_TRUE(tracker_->HasDeferredRequests());

  EXPECT_CALL(*mock_cm_, DestroyFinishedRequest(300)).Times(1);

  tracker_->OnDevicePutDone(0, 300);
  EXPECT_TRUE(tracker_->HasDeferredRequests());

  tracker_->OnDevicePutDone(1, 300);
  EXPECT_FALSE(tracker_->HasDeferredRequests());
}

TEST_F(PutTrackerTest, NoDeferWhenAllDoneBeforeReset) {
  auto req = MakeInferRequest(400, InferStage::kContext);
  tracker_->OnBatchDispatched({req});

  tracker_->OnDevicePutDone(0, 400);
  tracker_->OnDevicePutDone(1, 400);

  EXPECT_FALSE(tracker_->DeferDestructionIfNeeded(400));
  EXPECT_FALSE(tracker_->HasDeferredRequests());
}

TEST_F(PutTrackerTest, UnknownReqIdPutDoneIgnored) {
  tracker_->OnDevicePutDone(0, 999);
}

TEST_F(PutTrackerTest, MultiplePrefillRequests) {
  auto req1 = MakeInferRequest(500, InferStage::kContext);
  auto req2 = MakeInferRequest(501, InferStage::kContext);
  auto req3 = MakeInferRequest(502, InferStage::kDecode);
  tracker_->OnBatchDispatched({req1, req2, req3});

  EXPECT_TRUE(tracker_->DeferDestructionIfNeeded(500));
  EXPECT_TRUE(tracker_->DeferDestructionIfNeeded(501));
  EXPECT_FALSE(tracker_->DeferDestructionIfNeeded(502));

  EXPECT_CALL(*mock_cm_, DestroyFinishedRequest(500)).Times(1);
  EXPECT_CALL(*mock_cm_, DestroyFinishedRequest(501)).Times(1);

  tracker_->OnDevicePutDone(0, 500);
  tracker_->OnDevicePutDone(1, 500);
  tracker_->OnDevicePutDone(0, 501);
  tracker_->OnDevicePutDone(1, 501);

  EXPECT_FALSE(tracker_->HasDeferredRequests());
}

TEST_F(PutTrackerTest, DpCacheManagersUsedWhenSet) {
  auto dp_cm0 = std::make_shared<testing::NiceMock<MockCacheManager>>();
  auto dp_cm1 = std::make_shared<testing::NiceMock<MockCacheManager>>();
  tracker_->SetDpCacheManagers({dp_cm0, dp_cm1});

  auto req = MakeInferRequest(600, InferStage::kContext);
  tracker_->OnBatchDispatched({req});
  EXPECT_TRUE(tracker_->DeferDestructionIfNeeded(600, /*dp_idx=*/1));

  EXPECT_CALL(*dp_cm1, DestroyFinishedRequest(600)).Times(1);
  EXPECT_CALL(*dp_cm0, DestroyFinishedRequest(testing::_)).Times(0);
  EXPECT_CALL(*mock_cm_, DestroyFinishedRequest(testing::_)).Times(0);

  tracker_->OnDevicePutDone(0, 600);
  tracker_->OnDevicePutDone(1, 600);
}

TEST_F(PutTrackerTest, PartialDoneNoDefer) {
  auto req = MakeInferRequest(700, InferStage::kContext);
  tracker_->OnBatchDispatched({req});

  // 只有一个设备完成，pending 仍存在
  tracker_->OnDevicePutDone(0, 700);
  EXPECT_TRUE(tracker_->DeferDestructionIfNeeded(700));

  EXPECT_CALL(*mock_cm_, DestroyFinishedRequest(700)).Times(1);
  tracker_->OnDevicePutDone(1, 700);
  EXPECT_FALSE(tracker_->HasDeferredRequests());
}

TEST_F(PutTrackerTest, DpCacheManagersDpIdxOutOfRangeFallsBackToSingleManager) {
  // 设置了 dp_cache_managers_，但 dp_idx 超出其范围，应回退到 cache_manager_
  auto dp_cm0 = std::make_shared<testing::NiceMock<MockCacheManager>>();
  tracker_->SetDpCacheManagers({dp_cm0});  // 只有 1 个 dp manager，index 只有 0

  auto req = MakeInferRequest(800, InferStage::kContext);
  tracker_->OnBatchDispatched({req});

  // dp_idx=5 超出范围（dp_cache_managers_.size() == 1），应回退到 mock_cm_
  EXPECT_TRUE(tracker_->DeferDestructionIfNeeded(800, /*dp_idx=*/5));

  EXPECT_CALL(*mock_cm_, DestroyFinishedRequest(800)).Times(1);
  EXPECT_CALL(*dp_cm0, DestroyFinishedRequest(testing::_)).Times(0);

  tracker_->OnDevicePutDone(0, 800);
  tracker_->OnDevicePutDone(1, 800);
  EXPECT_FALSE(tracker_->HasDeferredRequests());
}

}  // namespace
}  // namespace ksana_llm
