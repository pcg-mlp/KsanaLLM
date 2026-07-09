/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "batch_scheduler/global_cache_fetcher.h"
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

// InferRequest 的 finished/output_tokens 等成员是对 Request 对象的引用，
// 必须保证 Request 的生命周期不短于 InferRequest。
std::vector<std::shared_ptr<Request>> g_alive_requests;

std::shared_ptr<InferRequest> MakeFetcherTestRequest(int64_t req_id, InferStage stage, int token_num = 10) {
  auto input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, req_ctx);
  g_alive_requests.push_back(request);
  auto infer_req = std::make_shared<InferRequest>(request, 0);
  infer_req->req_id = req_id;
  infer_req->infer_stage = stage;
  infer_req->block_token_num = 16;
  for (int i = 0; i < token_num; ++i) {
    infer_req->output_tokens.push_back(i + 100);
  }
  infer_req->ResetPrefillingTokens();
  return infer_req;
}

// ==================== GetAllDevicesContiguousPrefixLen 测试 ====================

class ContiguousPrefixLenTest : public ::testing::Test {
 protected:
  static size_t Compute(const std::vector<std::vector<int>>& ranges, size_t local_cached) {
    return GlobalCacheFetcher::GetAllDevicesContiguousPrefixLen(ranges, local_cached);
  }
};

TEST_F(ContiguousPrefixLenTest, EmptyDeviceRanges) { EXPECT_EQ(Compute({}, 0), 0u); }

TEST_F(ContiguousPrefixLenTest, SingleDeviceSingleRange) {
  // 设备 0 覆盖 [0, 9]，从位置 0 开始
  EXPECT_EQ(Compute({{0, 9}}, 0), 10u);
}

TEST_F(ContiguousPrefixLenTest, SingleDeviceWithLocalCachedOffset) {
  // 本地已缓存 5 个 token，设备覆盖 [5, 14]
  EXPECT_EQ(Compute({{5, 14}}, 5), 10u);
}

TEST_F(ContiguousPrefixLenTest, SingleDeviceGapAfterLocalCached) {
  // 本地已缓存 5 个，但设备从 7 开始（gap at pos 5-6）
  EXPECT_EQ(Compute({{7, 14}}, 5), 0u);
}

TEST_F(ContiguousPrefixLenTest, MultiDeviceIntersection) {
  // 设备 0: [0, 9]，设备 1: [0, 7]，交集连续前缀为 8
  EXPECT_EQ(Compute({{0, 9}, {0, 7}}, 0), 8u);
}

TEST_F(ContiguousPrefixLenTest, MultiDeviceWithGapOnOneDevice) {
  // 设备 0: [0, 9]，设备 1: [0, 3, 6, 9]（gap at 4,5）
  // 从 0 开始：min_end = min(9, 3) = 3，推进到 4
  // 设备 1 从 cursor 跳到 [6,9]，但 6 > 4，终止
  EXPECT_EQ(Compute({{0, 9}, {0, 3, 6, 9}}, 0), 4u);
}

TEST_F(ContiguousPrefixLenTest, MultiDeviceAllCover) {
  // 两个设备都完全覆盖 [10, 19]
  EXPECT_EQ(Compute({{10, 19}, {10, 19}}, 10), 10u);
}

TEST_F(ContiguousPrefixLenTest, ThreeDevicesMinimumRange) {
  // 设备 0: [0, 19], 设备 1: [0, 9], 设备 2: [0, 14]
  // 交集连续前缀 = min(19, 9, 14) + 1 = 10
  EXPECT_EQ(Compute({{0, 19}, {0, 9}, {0, 14}}, 0), 10u);
}

TEST_F(ContiguousPrefixLenTest, NoDeviceCoverPosition) {
  // 所有设备的区间从 5 开始，但 local_cached 是 0
  EXPECT_EQ(Compute({{5, 9}}, 0), 0u);
}

TEST_F(ContiguousPrefixLenTest, MultipleContiguousRanges) {
  // 单个设备的多段连续区间 [0,3] [4,7] [8,11]
  EXPECT_EQ(Compute({{0, 3, 4, 7, 8, 11}}, 0), 12u);
}

// ==================== GlobalCacheFetcher 同步模式测试 ====================

class FetcherSyncTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // ipc_func_wrapper = nullptr → 同步模式
    fetcher_ = std::make_unique<GlobalCacheFetcher>(/*device_count=*/2, /*ipc_func_wrapper=*/nullptr);
    fetcher_->Start();
  }

  void TearDown() override { fetcher_->Stop(); }

  std::unique_ptr<GlobalCacheFetcher> fetcher_;
};

TEST_F(FetcherSyncTest, IsAsyncReturnsFalseWithoutIpc) { EXPECT_FALSE(fetcher_->IsAsync()); }

TEST_F(FetcherSyncTest, SubmitRequestSyncCompletesImmediately) {
  auto req = MakeFetcherTestRequest(100, InferStage::kContext);
  fetcher_->SubmitRequest(req);

  auto completed = fetcher_->TakeCompletedRequests();
  ASSERT_EQ(completed.size(), 1u);
  EXPECT_EQ(completed[0]->req_id, 100);
}

TEST_F(FetcherSyncTest, HasPendingAfterSubmit) {
  // 同步模式下 SubmitRequest 后请求直接进入 completed 队列
  auto req = MakeFetcherTestRequest(200, InferStage::kContext);
  fetcher_->SubmitRequest(req);
  EXPECT_TRUE(fetcher_->HasPendingRequests());

  fetcher_->TakeCompletedRequests();
  EXPECT_FALSE(fetcher_->HasPendingRequests());
}

TEST_F(FetcherSyncTest, MultipleSubmitsAccumulate) {
  for (int i = 0; i < 5; ++i) {
    auto req = MakeFetcherTestRequest(300 + i, InferStage::kContext);
    fetcher_->SubmitRequest(req);
  }
  auto completed = fetcher_->TakeCompletedRequests();
  EXPECT_EQ(completed.size(), 5u);
}

TEST_F(FetcherSyncTest, TakeCompletedClearsList) {
  auto req = MakeFetcherTestRequest(400, InferStage::kContext);
  fetcher_->SubmitRequest(req);

  fetcher_->TakeCompletedRequests();
  auto second = fetcher_->TakeCompletedRequests();
  EXPECT_TRUE(second.empty());
}

TEST_F(FetcherSyncTest, TakeRejectedEmpty) {
  auto rejected = fetcher_->TakeRejectedRequests();
  EXPECT_TRUE(rejected.empty());
}

// ==================== OnDeviceComplete 测试 ====================

class FetcherAsyncSimTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 模拟 2 设备的异步模式，但不实际发 IPC
    fetcher_ = std::make_unique<GlobalCacheFetcher>(2, nullptr);
    fetcher_->Start();
  }

  void TearDown() override { fetcher_->Stop(); }

  // 手动将请求插入 pending 队列（模拟异步提交后的状态）
  void InsertPending(std::shared_ptr<InferRequest> req) {
    GlobalCacheFetcher::PendingRequest pending;
    pending.req = req;
    pending.devices_completed = 0;
    pending.total_devices = 2;
    fetcher_->pending_requests_[req->req_id] = std::move(pending);
  }

  std::unique_ptr<GlobalCacheFetcher> fetcher_;
};

TEST_F(FetcherAsyncSimTest, OnDeviceCompleteUnknownReqIgnored) {
  GlobalCacheQueryResult result;
  result.req_id = 999;
  result.device_id = 0;
  result.success = true;
  // 不应崩溃
  fetcher_->OnDeviceComplete(result);
}

TEST_F(FetcherAsyncSimTest, AllDevicesCompletedMovesToCompleted) {
  auto req = MakeFetcherTestRequest(600, InferStage::kContext, 20);
  InsertPending(req);

  // 设备 0 成功，覆盖 [0, 9]
  GlobalCacheQueryResult r0;
  r0.req_id = 600;
  r0.device_id = 0;
  r0.success = true;
  r0.matched_ranges = {0, 9};
  fetcher_->OnDeviceComplete(r0);
  EXPECT_TRUE(fetcher_->TakeCompletedRequests().empty());

  // 设备 1 成功，覆盖 [0, 9]
  GlobalCacheQueryResult r1;
  r1.req_id = 600;
  r1.device_id = 1;
  r1.success = true;
  r1.matched_ranges = {0, 9};
  fetcher_->OnDeviceComplete(r1);

  auto completed = fetcher_->TakeCompletedRequests();
  EXPECT_EQ(completed.size(), 1u);
}

TEST_F(FetcherAsyncSimTest, OneDeviceFailedStillCompletes) {
  auto req = MakeFetcherTestRequest(700, InferStage::kContext, 20);
  InsertPending(req);

  GlobalCacheQueryResult r0;
  r0.req_id = 700;
  r0.device_id = 0;
  r0.success = false;
  fetcher_->OnDeviceComplete(r0);

  GlobalCacheQueryResult r1;
  r1.req_id = 700;
  r1.device_id = 1;
  r1.success = true;
  r1.matched_ranges = {0, 9};
  fetcher_->OnDeviceComplete(r1);

  auto completed = fetcher_->TakeCompletedRequests();
  EXPECT_EQ(completed.size(), 1u);
}

TEST_F(FetcherAsyncSimTest, CachedTokensCollectedFromFirstDevice) {
  auto req = MakeFetcherTestRequest(800, InferStage::kContext, 10);
  InsertPending(req);

  GlobalCacheQueryResult r0;
  r0.req_id = 800;
  r0.device_id = 0;
  r0.success = true;
  r0.matched_ranges = {0, 9};
  r0.cached_tokens = {42, 43};
  r0.has_cached_tokens = true;
  fetcher_->OnDeviceComplete(r0);

  // 验证 pending 状态已记录 cached_tokens
  auto& pending = fetcher_->pending_requests_[800];
  EXPECT_TRUE(pending.has_cached_tokens);
  EXPECT_EQ(pending.cached_tokens, (std::vector<int>{42, 43}));

  GlobalCacheQueryResult r1;
  r1.req_id = 800;
  r1.device_id = 1;
  r1.matched_ranges = {0, 9};
  r1.success = true;
  fetcher_->OnDeviceComplete(r1);
}

TEST_F(FetcherAsyncSimTest, CompletionCallbackFired) {
  int64_t cb_req_id = -1;
  bool cb_success = false;
  fetcher_->SetCompletionCallback([&](int64_t req_id, bool success) {
    cb_req_id = req_id;
    cb_success = success;
  });

  auto req = MakeFetcherTestRequest(900, InferStage::kContext, 20);
  InsertPending(req);

  GlobalCacheQueryResult r0;
  r0.req_id = 900;
  r0.device_id = 0;
  r0.matched_ranges = {0, 9};
  r0.success = true;
  fetcher_->OnDeviceComplete(r0);
  GlobalCacheQueryResult r1;
  r1.req_id = 900;
  r1.device_id = 1;
  r1.matched_ranges = {0, 9};
  r1.success = true;
  fetcher_->OnDeviceComplete(r1);

  EXPECT_EQ(cb_req_id, 900);
  EXPECT_TRUE(cb_success);
}

// ==================== UpdateRequestWorkload 测试 ====================

class UpdateWorkloadTest : public ::testing::Test {
 protected:
  void SetUp() override { fetcher_ = std::make_unique<GlobalCacheFetcher>(1, nullptr); }

  void CallUpdate(std::shared_ptr<InferRequest>& req, size_t new_cached_len, const std::vector<int>& cached_tokens = {},
                  bool has_cached_tokens = false) {
    fetcher_->UpdateRequestWorkload(req, new_cached_len, cached_tokens, has_cached_tokens);
  }

  std::unique_ptr<GlobalCacheFetcher> fetcher_;
};

TEST_F(UpdateWorkloadTest, ZeroCachedLenNoOp) {
  auto req = MakeFetcherTestRequest(1000, InferStage::kContext, 20);
  auto workload_before = req->GetPlanningWorkload();
  CallUpdate(req, 0);
  EXPECT_EQ(req->GetPlanningWorkload().prefill_token_num, workload_before.prefill_token_num);
}

TEST_F(UpdateWorkloadTest, PartialHitUpdatesWorkload) {
  auto req = MakeFetcherTestRequest(1100, InferStage::kContext, 20);
  // 初始 workload: prefill_start_offset=0, prefill_token_num=20
  CallUpdate(req, 10);

  auto planning = req->GetPlanningWorkload();
  EXPECT_EQ(planning.prefill_start_offset, 10u);
  EXPECT_EQ(planning.prefill_token_num, 10u);

  auto remaining = req->GetRemainingWorkload();
  EXPECT_EQ(remaining.prefill_start_offset, 10u);
  EXPECT_EQ(remaining.prefill_token_num, 10u);
}

TEST_F(UpdateWorkloadTest, FullHitWithoutTokensSwitchesToDecode) {
  auto req = MakeFetcherTestRequest(1200, InferStage::kContext, 20);
  CallUpdate(req, 20);

  EXPECT_EQ(req->infer_stage, InferStage::kDecode);
  auto planning = req->GetPlanningWorkload();
  // full hit without tokens: prefill 最后 1 个 token
  EXPECT_EQ(planning.prefill_start_offset, 19u);
  EXPECT_EQ(planning.prefill_token_num, 1u);
}

TEST_F(UpdateWorkloadTest, FullHitWithTokensAppends) {
  auto req = MakeFetcherTestRequest(1300, InferStage::kContext, 20);
  size_t output_before = req->output_tokens.size();
  CallUpdate(req, 20, {42, 43}, true);

  EXPECT_EQ(req->infer_stage, InferStage::kDecode);
  EXPECT_EQ(req->output_tokens.size(), output_before + 2);
  auto planning = req->GetPlanningWorkload();
  EXPECT_EQ(planning.prefill_start_offset, 20u);
  EXPECT_EQ(planning.prefill_token_num, 2u);
}

// ==================== PD-disagg Prefill full-hit 测试 ====================

class PdDisaggPrefillUpdateWorkloadTest : public ::testing::Test {
 protected:
  void SetUp() override { fetcher_ = std::make_unique<GlobalCacheFetcher>(1, nullptr, /*role=*/"pd_v2_prefill"); }

  void CallUpdate(std::shared_ptr<InferRequest>& req, size_t new_cached_len, const std::vector<int>& cached_tokens = {},
                  bool has_cached_tokens = false) {
    fetcher_->UpdateRequestWorkload(req, new_cached_len, cached_tokens, has_cached_tokens);
  }

  std::unique_ptr<GlobalCacheFetcher> fetcher_;
};

TEST_F(PdDisaggPrefillUpdateWorkloadTest, FullHitStaysContextAndRetainsLastToken) {
  auto req = MakeFetcherTestRequest(1400, InferStage::kContext, 20);
  CallUpdate(req, 20, {42, 43}, true);

  EXPECT_EQ(req->infer_stage, InferStage::kContext);
  auto planning = req->GetPlanningWorkload();
  EXPECT_EQ(planning.prefill_start_offset, 19u);
  EXPECT_EQ(planning.prefill_token_num, 1u);
  EXPECT_EQ(req->output_tokens.size(), 20u);
}

// ==================== AppendCachedTokens 测试 ====================

TEST(AppendCachedTokensTest, AppendsToOutputAndPrefilling) {
  auto req = MakeFetcherTestRequest(1600, InferStage::kContext, 10);
  size_t output_before = req->output_tokens.size();
  size_t prefilling_before = req->GetPrefillingTokens().size();

  req->AppendCachedTokens({42, 43, 44});

  EXPECT_EQ(req->output_tokens.size(), output_before + 3);
  EXPECT_EQ(req->GetPrefillingTokens().size(), prefilling_before + 3);
  EXPECT_EQ(req->output_tokens.back(), 44);
}

TEST(AppendCachedTokensTest, EmptyTokensNoOp) {
  auto req = MakeFetcherTestRequest(1700, InferStage::kContext, 5);
  size_t output_before = req->output_tokens.size();

  req->AppendCachedTokens({});

  EXPECT_EQ(req->output_tokens.size(), output_before);
}

// ==================== SetPlanningWorkload/SetRemainingWorkload 测试 ==

TEST(WorkloadSetterTest, SetPlanningWorkload) {
  auto req = MakeFetcherTestRequest(1800, InferStage::kContext, 10);
  ScheduleTaskWorkload wl;
  wl.prefill_start_offset = 5;
  wl.prefill_token_num = 5;

  req->SetPlanningWorkload(wl);
  auto got = req->GetPlanningWorkload();
  EXPECT_EQ(got.prefill_start_offset, 5u);
  EXPECT_EQ(got.prefill_token_num, 5u);
}

TEST(WorkloadSetterTest, SetRemainingWorkload) {
  auto req = MakeFetcherTestRequest(1900, InferStage::kContext, 10);
  ScheduleTaskWorkload wl;
  wl.prefill_start_offset = 3;
  wl.prefill_token_num = 7;

  req->SetRemainingWorkload(wl);
  auto got = req->GetRemainingWorkload();
  EXPECT_EQ(got.prefill_start_offset, 3u);
  EXPECT_EQ(got.prefill_token_num, 7u);
}

// ==================== Legacy full-hit（非 PD-disagg Prefill）====================

class LegacyFullHitUpdateWorkloadTest : public ::testing::Test {
 protected:
  void SetUp() override { fetcher_ = std::make_unique<GlobalCacheFetcher>(1, nullptr, /*role=*/""); }

  void CallUpdate(std::shared_ptr<InferRequest>& req, size_t new_cached_len, const std::vector<int>& cached_tokens = {},
                  bool has_cached_tokens = false) {
    fetcher_->UpdateRequestWorkload(req, new_cached_len, cached_tokens, has_cached_tokens);
  }

  std::unique_ptr<GlobalCacheFetcher> fetcher_;
};

TEST_F(LegacyFullHitUpdateWorkloadTest, FullHitWithTokensSwitchesToDecode) {
  // 全命中 + tokens 就绪 → 切换 kDecode，workload 正确
  auto req = MakeFetcherTestRequest(2000, InferStage::kContext, 10);
  CallUpdate(req, /*new_cached_len=*/10, /*cached_tokens=*/{42, 43}, /*has_cached_tokens=*/true);

  EXPECT_FALSE(req->finished);
  EXPECT_EQ(req->infer_stage, InferStage::kDecode);
  auto planning = req->GetPlanningWorkload();
  EXPECT_EQ(planning.prefill_start_offset, 10u);
  EXPECT_EQ(planning.prefill_token_num, 2u);
}

TEST_F(LegacyFullHitUpdateWorkloadTest, FullHitWithoutTokensSwitchesToDecode) {
  auto req = MakeFetcherTestRequest(2001, InferStage::kContext, 10);
  CallUpdate(req, /*new_cached_len=*/10, /*cached_tokens=*/{}, /*has_cached_tokens=*/false);

  EXPECT_FALSE(req->finished);
  EXPECT_EQ(req->infer_stage, InferStage::kDecode);
}

}  // namespace
}  // namespace ksana_llm
