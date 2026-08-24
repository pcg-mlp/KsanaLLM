/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/global_cache_query_hook.h"
#include "batch_scheduler/event_driven_scheduler/schedule_context.h"
#include "batch_scheduler/global_cache_fetcher.h"
#include "cache_manager/cache_manager_interface.h"
#include "configure/environment.h"
#include "loguru.hpp"
#include "runtime/infer_request.h"
#include "utils/request.h"
#include "utils/singleton.h"

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
  MOCK_METHOD(Status, DropUnbackedSwaBlocks, (int64_t, size_t, std::vector<int>&, std::vector<size_t>&), (override));
  MOCK_METHOD(void, BeginSwaPutHold, (int64_t), (override));
  MOCK_METHOD(void, EndSwaPutHold, (int64_t), (override));
  MOCK_METHOD(void, DestroyFinishedRequest, (int64_t), (override));
  MOCK_METHOD(bool, IsPrefixCachingEnabled, (), (override));
};

// InferRequest 持有 Request 成员的引用，必须保证 Request 生命周期不短于 InferRequest。
std::vector<std::shared_ptr<Request>> g_alive_requests;

std::shared_ptr<InferRequest> MakeHookTestRequest(int64_t req_id, int token_num = 10) {
  auto input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, req_ctx);
  g_alive_requests.push_back(request);
  auto infer_req = std::make_shared<InferRequest>(request, 0);
  infer_req->req_id = req_id;
  infer_req->infer_stage = InferStage::kContext;
  infer_req->block_token_num = 16;
  infer_req->attn_dp_group_id = 0;
  for (int i = 0; i < token_num; ++i) {
    infer_req->output_tokens.push_back(i + 100);
  }
  infer_req->ResetPrefillingTokens();
  return infer_req;
}

class QueryHookTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_cm_ = std::make_shared<testing::NiceMock<MockCacheManager>>();

    // 默认 mock 行为：prefix cache 无命中，block 分配成功
    ON_CALL(*mock_cm_,
            GetRequestPrefixBlockNumber(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillByDefault(testing::Return(Status()));
    ON_CALL(*mock_cm_, GetRequestStepBlockNumber(testing::_, testing::_, testing::_)).WillByDefault(testing::Return(2));
    ON_CALL(*mock_cm_, AllocateRequestBlocks(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillByDefault(testing::Return(Status()));

    // 同步模式的 fetcher（无 IPC）
    fetcher_ = std::make_shared<GlobalCacheFetcher>(1, nullptr);
    fetcher_->Start();

    BatchSchedulerConfig config;
    hook_ = std::make_unique<GlobalCacheQueryHook>(fetcher_, config, /*hook_idx=*/0);

    dp_cache_managers_ = {mock_cm_};
    context_.SetDpCacheManagers(&dp_cache_managers_);
  }

  void TearDown() override { fetcher_->Stop(); }

  std::shared_ptr<testing::NiceMock<MockCacheManager>> mock_cm_;
  std::shared_ptr<GlobalCacheFetcher> fetcher_;
  std::unique_ptr<GlobalCacheQueryHook> hook_;
  ScheduleContext context_;
  std::vector<std::shared_ptr<CacheManagerInterface>> dp_cache_managers_;
};

TEST_F(QueryHookTest, EmptyRequestsReturnEmpty) {
  auto result = hook_->Process({}, context_);
  EXPECT_TRUE(result.empty());
}

TEST_F(QueryHookTest, SyncModeProcessReturnsContinue) {
  auto req = MakeHookTestRequest(100);
  auto result = hook_->Process({req}, context_);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[100].action, NextStepAction::kContinue);
}

TEST_F(QueryHookTest, BlockAllocFailReturnsContinue) {
  ON_CALL(*mock_cm_, AllocateRequestBlocks(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault(testing::Return(Status(RET_RUNTIME_FAILED, "out of blocks")));

  auto req = MakeHookTestRequest(200);
  auto result = hook_->Process({req}, context_);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[200].action, NextStepAction::kContinue);
}

TEST_F(QueryHookTest, PrefixCacheHitUpdatesWorkload) {
  ON_CALL(*mock_cm_,
          GetRequestPrefixBlockNumber(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault([](int64_t, const std::vector<int>&, size_t, size_t& shared_block_num, size_t& req_step_block_num,
                        size_t& shared_token_num) {
        shared_block_num = 1;
        shared_token_num = 5;
        req_step_block_num = 2;
        return Status();
      });

  auto req = MakeHookTestRequest(300, 20);
  hook_->Process({req}, context_);

  auto workload = req->GetPlanningWorkload();
  EXPECT_EQ(workload.prefill_start_offset, 5u);
}

TEST_F(QueryHookTest, UpdateAsyncResultEmptyInput) {
  auto result = hook_->UpdateAsyncResult({}, context_);
  EXPECT_TRUE(result.empty());
}

TEST_F(QueryHookTest, UpdateAsyncResultCompletedRequests) {
  // 先提交一个请求（同步模式立即完成）
  auto req = MakeHookTestRequest(400);
  hook_->Process({req}, context_);

  // 模拟 async result（实际在同步模式下不会到这里，但验证逻辑正确性）
  AsyncTaskResult task_result;
  task_result.hook_idx = 0;
  auto result = hook_->UpdateAsyncResult({task_result}, context_);
  // 同步模式下 completed 已被 Process 取走，这里为空
  EXPECT_TRUE(result.empty());
}

TEST_F(QueryHookTest, MultipleRequestsProcessed) {
  auto req1 = MakeHookTestRequest(500, 15);
  auto req2 = MakeHookTestRequest(501, 20);
  auto result = hook_->Process({req1, req2}, context_);
  EXPECT_EQ(result.size(), 2u);
  EXPECT_EQ(result[500].action, NextStepAction::kContinue);
  EXPECT_EQ(result[501].action, NextStepAction::kContinue);
}

TEST_F(QueryHookTest, SetAsyncTaskFinishedCallback) {
  bool callback_fired = false;
  hook_->SetAsyncTaskFinishedCallback([&](const EventAsyncTaskFinishedInfo& info) {
    callback_fired = true;
    EXPECT_FALSE(info.task_results.empty());
    EXPECT_EQ(info.task_results[0].hook_idx, 0);
  });

  // 手动模拟一个请求完成（插入 pending 再 complete）
  auto req = MakeHookTestRequest(600);
  GlobalCacheFetcher::PendingRequest pending;
  pending.req = req;
  pending.devices_completed = 0;
  pending.total_devices = 1;
  fetcher_->pending_requests_[600] = std::move(pending);

  GlobalCacheQueryResult r0;
  r0.req_id = 600;
  r0.device_id = 0;
  r0.success = true;
  fetcher_->OnDeviceComplete(r0);
  EXPECT_TRUE(callback_fired);
}

TEST_F(QueryHookTest, UpdateAsyncResultWithActualCompletedRequests) {
  // 直接插入 pending + 触发完成，跳过 Process（不提前取走），
  // 验证 UpdateAsyncResult 能从 completed 队列取出请求并返回 kContinue。
  auto req = MakeHookTestRequest(700);
  GlobalCacheFetcher::PendingRequest pending;
  pending.req = req;
  pending.devices_completed = 0;
  pending.total_devices = 1;
  fetcher_->pending_requests_[700] = std::move(pending);

  // 空 matched_ranges + standalone 模式：UpdateRequestWorkload 不操作（new_cached_len=0），
  // req 不被拒绝，进入 completed_requests_。
  GlobalCacheQueryResult r0;
  r0.req_id = 700;
  r0.device_id = 0;
  r0.success = true;
  r0.matched_ranges = {};
  fetcher_->OnDeviceComplete(r0);

  AsyncTaskResult task_result;
  task_result.hook_idx = 0;
  auto result = hook_->UpdateAsyncResult({task_result}, context_);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[700].action, NextStepAction::kContinue);
}

// ==================== PD-disagg Prefill Hook 测试 ====================

class QueryHookPdDisaggPrefillTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_cm_ = std::make_shared<testing::NiceMock<MockCacheManager>>();

    ON_CALL(*mock_cm_,
            GetRequestPrefixBlockNumber(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillByDefault(testing::Return(Status()));
    ON_CALL(*mock_cm_, GetRequestStepBlockNumber(testing::_, testing::_, testing::_)).WillByDefault(testing::Return(2));
    ON_CALL(*mock_cm_, AllocateRequestBlocks(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillByDefault(testing::Return(Status()));

    fetcher_ = std::make_shared<GlobalCacheFetcher>(1, nullptr, /*role=*/"pd_v2_prefill");
    fetcher_->Start();

    BatchSchedulerConfig config;
    hook_ = std::make_unique<GlobalCacheQueryHook>(fetcher_, config, /*hook_idx=*/0);

    dp_cache_managers_ = {mock_cm_};
    context_.SetDpCacheManagers(&dp_cache_managers_);
  }

  void TearDown() override { fetcher_->Stop(); }

  std::shared_ptr<testing::NiceMock<MockCacheManager>> mock_cm_;
  std::shared_ptr<GlobalCacheFetcher> fetcher_;
  std::unique_ptr<GlobalCacheQueryHook> hook_;
  ScheduleContext context_;
  std::vector<std::shared_ptr<CacheManagerInterface>> dp_cache_managers_;
};

TEST_F(QueryHookPdDisaggPrefillTest, FullGlobalHitStaysContext) {
  auto req = MakeHookTestRequest(800, 20);
  GlobalCacheFetcher::PendingRequest pending;
  pending.req = req;
  pending.devices_completed = 0;
  pending.total_devices = 1;
  fetcher_->pending_requests_[800] = std::move(pending);

  GlobalCacheQueryResult r0;
  r0.req_id = 800;
  r0.device_id = 0;
  r0.success = true;
  r0.matched_ranges = {0, 19};
  r0.cached_tokens = {42};
  r0.has_cached_tokens = true;
  fetcher_->OnDeviceComplete(r0);

  AsyncTaskResult task_result;
  task_result.hook_idx = 0;
  auto result = hook_->UpdateAsyncResult({task_result}, context_);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[800].action, NextStepAction::kContinue);

  // UpdateAsyncResult already drained completed_requests_ via TakeCompletedRequests
  // (see GlobalCacheQueryHook::UpdateAsyncResult), so assert on the req object
  // directly instead of taking from the fetcher a second time (mirrors the
  // UpdateAsyncResultWithActualCompletedRequests pattern). A full global cache
  // hit on a pd_v2 Prefill node must keep the request in Context: Prefill still
  // has to run to RDMA the KV back to its Decode peer.
  EXPECT_EQ(req->infer_stage, InferStage::kContext);
}

// Local full prefix-cache hit on a pd_v2 Prefill node must NOT flip to decode.
TEST_F(QueryHookPdDisaggPrefillTest, LocalFullPrefixHitStaysContext) {
  ON_CALL(*mock_cm_,
          GetRequestPrefixBlockNumber(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault([](int64_t, const std::vector<int>&, size_t seq, size_t& shared_block_num,
                        size_t& req_step_block_num, size_t& shared_token_num) {
        shared_block_num = 2;
        req_step_block_num = 1;
        shared_token_num = seq;  // full hit
        return Status();
      });
  auto req = MakeHookTestRequest(810, 12);
  hook_->Process({req}, context_);
  EXPECT_EQ(req->infer_stage, InferStage::kContext);  // prefill node never flips
}

// ==================== Local full prefix-cache hit (standalone) ====================
// LocalFullPrefixHitFlipsToDecode 已删除：prefix 查询硬上限 seq-1，不再 FullHit flip decode。
// 覆盖见文件末尾 LocalPrefixHitUsesPartialWorkload。

// ==================== pd_v2 Decode admission-gate (defer mode) ====================

class QueryHookDeferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_cm_ = std::make_shared<testing::NiceMock<MockCacheManager>>();
    ON_CALL(*mock_cm_,
            GetRequestPrefixBlockNumber(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillByDefault(testing::Return(Status()));
    ON_CALL(*mock_cm_, GetRequestStepBlockNumber(testing::_, testing::_, testing::_)).WillByDefault(testing::Return(2));
    ON_CALL(*mock_cm_, AllocateRequestBlocks(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillByDefault(testing::Return(Status()));
    // Plenty of usable blocks by default; individual tests lower this to force a defer.
    ON_CALL(*mock_cm_, GetUsableBlockNumber()).WillByDefault(testing::Return(1000));

    fetcher_ = std::make_shared<GlobalCacheFetcher>(1, nullptr);
    fetcher_->Start();

    BatchSchedulerConfig config;
    config.max_batch_size = 2;  // max_inflight_reqs_ = 4
    hook_ = std::make_unique<GlobalCacheQueryHook>(fetcher_, config, /*hook_idx=*/0,
                                                   /*defer_admission_on_block_full=*/true);
    dp_cache_managers_ = {mock_cm_};
    context_.SetDpCacheManagers(&dp_cache_managers_);
  }
  void TearDown() override { fetcher_->Stop(); }

  std::shared_ptr<testing::NiceMock<MockCacheManager>> mock_cm_;
  std::shared_ptr<GlobalCacheFetcher> fetcher_;
  std::unique_ptr<GlobalCacheQueryHook> hook_;
  ScheduleContext context_;
  std::vector<std::shared_ptr<CacheManagerInterface>> dp_cache_managers_;
};

TEST_F(QueryHookDeferTest, AdmitSucceedsWhenBlocksAvailable) {
  auto req = MakeHookTestRequest(1001);
  auto result = hook_->Process({req}, context_);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[1001].action, NextStepAction::kContinue);  // sync fetcher completes immediately
  EXPECT_EQ(hook_->DeferredCountForTest(), 0u);
}

TEST_F(QueryHookDeferTest, InflightCapDefersBeyondLimit) {
  // max_inflight_reqs_ = 4. Admit 4, then the 5th is deferred purely by the cap.
  std::vector<std::shared_ptr<InferRequest>> admitted;
  for (int i = 0; i < 4; ++i) admitted.push_back(MakeHookTestRequest(2000 + i));
  hook_->Process(admitted, context_);
  EXPECT_EQ(hook_->DeferredCountForTest(), 0u);

  auto over = MakeHookTestRequest(2099);
  auto result = hook_->Process({over}, context_);
  EXPECT_EQ(result[2099].action, NextStepAction::kWaitAsyncResult);
  EXPECT_EQ(hook_->DeferredCountForTest(), 1u);
}

TEST_F(QueryHookDeferTest, NotifyBlocksFreedFiresSentinelWhenPending) {
  bool fired = false;
  int seen_task_id = 0;
  hook_->SetAsyncTaskFinishedCallback([&](const EventAsyncTaskFinishedInfo& info) {
    fired = true;
    ASSERT_FALSE(info.task_results.empty());
    seen_task_id = info.task_results[0].task_id;
  });
  // Force one parked request via in-flight cap.
  std::vector<std::shared_ptr<InferRequest>> admitted;
  for (int i = 0; i < 4; ++i) {
    admitted.push_back(MakeHookTestRequest(3000 + i));
  }
  hook_->Process(admitted, context_);
  hook_->Process({MakeHookTestRequest(3099)}, context_);
  ASSERT_EQ(hook_->DeferredCountForTest(), 1u);

  hook_->OnEvent(HookEvent(HookEventType::kBlocksFreed));
  EXPECT_TRUE(fired);
  EXPECT_EQ(seen_task_id, -1);  // sentinel
}

TEST_F(QueryHookDeferTest, NotifyBlocksFreedNoopWhenEmpty) {
  bool fired = false;
  hook_->SetAsyncTaskFinishedCallback([&](const EventAsyncTaskFinishedInfo&) { fired = true; });
  hook_->OnEvent(HookEvent(HookEventType::kBlocksFreed));  // nothing pending -> no-op
  EXPECT_FALSE(fired);
}

TEST_F(QueryHookDeferTest, OnRequestExitedReleasesSlotAndWakes) {
  bool fired = false;
  hook_->SetAsyncTaskFinishedCallback([&](const EventAsyncTaskFinishedInfo&) { fired = true; });
  std::vector<std::shared_ptr<InferRequest>> admitted;
  for (int i = 0; i < 4; ++i) {
    admitted.push_back(MakeHookTestRequest(4000 + i));
  }
  hook_->Process(admitted, context_);
  hook_->Process({MakeHookTestRequest(4099)}, context_);
  ASSERT_EQ(hook_->DeferredCountForTest(), 1u);

  hook_->OnEvent(HookEvent(HookEventType::kRequestExited, 4000));  // erases slot -> wake deferred admission
  EXPECT_TRUE(fired);
}

TEST_F(QueryHookDeferTest, UpdateAsyncResultDrainsAndAdmits) {
  std::vector<std::shared_ptr<InferRequest>> admitted;
  for (int i = 0; i < 4; ++i) {
    admitted.push_back(MakeHookTestRequest(5000 + i));
  }
  hook_->Process(admitted, context_);
  hook_->Process({MakeHookTestRequest(5099)}, context_);
  ASSERT_EQ(hook_->DeferredCountForTest(), 1u);

  hook_->OnEvent(HookEvent(HookEventType::kRequestExited, 5000));
  AsyncTaskResult tr;
  tr.hook_idx = 0;
  tr.task_id = -1;
  auto result = hook_->UpdateAsyncResult({tr}, context_);
  EXPECT_EQ(hook_->DeferredCountForTest(), 0u);  // drained + admitted
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[5099].action, NextStepAction::kContinue);  // sync fetcher completed it
}

TEST_F(QueryHookDeferTest, UpdateAsyncResultRetryStillFailsRequeues) {
  std::vector<std::shared_ptr<InferRequest>> admitted;
  for (int i = 0; i < 4; ++i) {
    admitted.push_back(MakeHookTestRequest(5100 + i));
  }
  hook_->Process(admitted, context_);
  hook_->Process({MakeHookTestRequest(5199)}, context_);
  ASSERT_EQ(hook_->DeferredCountForTest(), 1u);

  AsyncTaskResult tr;
  tr.hook_idx = 0;
  tr.task_id = -1;
  hook_->UpdateAsyncResult({tr}, context_);  // inflight cap still full -> requeued
  EXPECT_EQ(hook_->DeferredCountForTest(), 1u);
}

TEST_F(QueryHookDeferTest, UpdateAsyncResultAllocFailRequeues) {
  ON_CALL(*mock_cm_, AllocateRequestBlocks(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault(testing::Return(Status(RET_RUNTIME_FAILED, "oom")));
  auto result = hook_->Process({MakeHookTestRequest(5200)}, context_);
  EXPECT_EQ(result[5200].action, NextStepAction::kWaitAsyncResult);
  ASSERT_EQ(hook_->DeferredCountForTest(), 1u);

  AsyncTaskResult tr;
  tr.hook_idx = 0;
  tr.task_id = -1;
  hook_->UpdateAsyncResult({tr}, context_);
  EXPECT_EQ(hook_->DeferredCountForTest(), 1u);
}

TEST_F(QueryHookDeferTest, AllocFailureDefersInDeferMode) {
  // Block-usable check passes, but AllocateRequestBlocks fails -> deferred (not fallback).
  ON_CALL(*mock_cm_, AllocateRequestBlocks(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault(testing::Return(Status(RET_RUNTIME_FAILED, "oom")));
  auto result = hook_->Process({MakeHookTestRequest(6000)}, context_);
  EXPECT_EQ(result[6000].action, NextStepAction::kWaitAsyncResult);
  EXPECT_EQ(hook_->DeferredCountForTest(), 1u);
}

// kRequestExited event on a non-defer hook is a no-op (early return).
TEST_F(QueryHookTest, OnRequestExitedNoopWhenNotDeferMode) {
  hook_->OnEvent(HookEvent(HookEventType::kRequestExited, 12345));  // defer_admission_on_block_full_ == false
  SUCCEED();
}

// ==================== Local prefix-cache hit (standalone) ====================

TEST_F(QueryHookTest, LocalPrefixHitUsesPartialWorkload) {
  ON_CALL(*mock_cm_,
          GetRequestPrefixBlockNumber(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault([](int64_t, const std::vector<int>&, size_t seq, size_t& shared_block_num,
                        size_t& req_step_block_num, size_t& shared_token_num) {
        shared_block_num = 2;
        req_step_block_num = 1;
        shared_token_num = seq > 0 ? seq - 1 : 0;  // 硬上限 seq-1，不再 FullHit flip decode
        return Status();
      });
  auto req = MakeHookTestRequest(910, 12);
  hook_->Process({req}, context_);
  EXPECT_EQ(req->infer_stage, InferStage::kContext);
  EXPECT_EQ(req->GetPlanningWorkload().prefill_start_offset, 11u);
  EXPECT_EQ(req->GetPlanningWorkload().prefill_token_num, 1u);
}

}  // namespace
}  // namespace ksana_llm
