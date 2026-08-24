/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "batch_scheduler/global_cache_fetcher.h"
#include "ipc/ipc_func_wrapper.h"
#include "loguru.hpp"
#include "profiler/timer.h"
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

// ==================== Prefill 角色测试 ====================

TEST(FetcherPrefillRole, SetsMaxNewTokensToOne) {
  auto fetcher = std::make_unique<GlobalCacheFetcher>(1, nullptr, "prefill");
  fetcher->Start();

  auto req = MakeFetcherTestRequest(500, InferStage::kContext);
  req->sampling_config.max_new_tokens = 100;
  fetcher->SubmitRequest(req);

  auto completed = fetcher->TakeCompletedRequests();
  ASSERT_EQ(completed.size(), 1u);
  EXPECT_EQ(completed[0]->sampling_config.max_new_tokens, 1u);
  fetcher->Stop();
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

TEST_F(UpdateWorkloadTest, ZeroCachedLenNonDecodeNoOp) {
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

TEST_F(UpdateWorkloadTest, FullHitWithoutTokensCapsToSeqMinusOne) {
  auto req = MakeFetcherTestRequest(1200, InferStage::kContext, 20);
  CallUpdate(req, 20);

  // 无 tokens：命中硬上限 seq-1，不 flip decode。
  EXPECT_EQ(req->infer_stage, InferStage::kContext);
  auto planning = req->GetPlanningWorkload();
  EXPECT_EQ(planning.prefill_start_offset, 19u);
  EXPECT_EQ(planning.prefill_token_num, 1u);
}

TEST_F(UpdateWorkloadTest, SwaFullHitWithoutTokensBlockAligns) {
  // SWA + GlobalCache 全覆盖无 tokens：seq-1 后再按 block 下取整（同旧 FullHit）。
  auto req = MakeFetcherTestRequest(1201, InferStage::kContext, 20);
  req->swa_kv_cache_blocks = {1};  // 非空即视为 SWA
  req->block_token_num = 16;
  CallUpdate(req, 20);

  EXPECT_EQ(req->infer_stage, InferStage::kContext);
  auto planning = req->GetPlanningWorkload();
  // usable = min(20, 19) = 19 → floor(19/16)*16 = 16
  EXPECT_EQ(planning.prefill_start_offset, 16u);
  EXPECT_EQ(planning.prefill_token_num, 4u);
}

TEST_F(UpdateWorkloadTest, SwaFullHitAlignedLeavesOneBlock) {
  auto req = MakeFetcherTestRequest(1202, InferStage::kContext, 16);
  req->swa_kv_cache_blocks = {1};
  req->block_token_num = 16;
  CallUpdate(req, 16);

  EXPECT_EQ(req->infer_stage, InferStage::kContext);
  // usable = 15 → floor = 0，不更新 workload（仍为初始全量 prefill）
  auto planning = req->GetPlanningWorkload();
  EXPECT_EQ(planning.prefill_start_offset, 0u);
  EXPECT_EQ(planning.prefill_token_num, 16u);
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

// ==================== PD-disagg Prefill（seq-1 硬上限，不 flip kDecode）====================

class PdDisaggPrefillUpdateWorkloadTest : public ::testing::Test {
 protected:
  void SetUp() override { fetcher_ = std::make_unique<GlobalCacheFetcher>(1, nullptr, /*role=*/"pd_v2_prefill"); }

  void CallUpdate(std::shared_ptr<InferRequest>& req, size_t new_cached_len, const std::vector<int>& cached_tokens = {},
                  bool has_cached_tokens = false) {
    fetcher_->UpdateRequestWorkload(req, new_cached_len, cached_tokens, has_cached_tokens);
  }

  std::unique_ptr<GlobalCacheFetcher> fetcher_;
};

TEST_F(PdDisaggPrefillUpdateWorkloadTest, RemoteFullCoverStaysContextCapsToSeqMinusOne) {
  auto req = MakeFetcherTestRequest(1400, InferStage::kContext, 20);
  CallUpdate(req, 20, {42, 43}, true);

  EXPECT_EQ(req->infer_stage, InferStage::kContext);
  auto planning = req->GetPlanningWorkload();
  EXPECT_EQ(planning.prefill_start_offset, 19u);
  EXPECT_EQ(planning.prefill_token_num, 1u);
  EXPECT_EQ(req->output_tokens.size(), 20u);
}

// ==================== Decode 角色 Reject 测试 ====================

TEST(FetcherDecodeRole, RejectsWhenNotFullHit) {
  auto fetcher = std::make_unique<GlobalCacheFetcher>(1, nullptr, "decode");
  fetcher->Start();

  auto req = MakeFetcherTestRequest(1400, InferStage::kContext, 20);
  fetcher->SubmitRequest(req);

  auto completed = fetcher->TakeCompletedRequests();
  EXPECT_TRUE(completed.empty());

  auto rejected = fetcher->TakeRejectedRequests();
  ASSERT_EQ(rejected.size(), 1u);
  EXPECT_TRUE(rejected[0]->finished);
  fetcher->Stop();
}

TEST(FetcherDecodeRole, RejectsFullHitWithoutTokens) {
  auto fetcher = std::make_unique<GlobalCacheFetcher>(1, nullptr, "decode");
  auto req = MakeFetcherTestRequest(1500, InferStage::kContext, 10);

  // 手动插入 pending 并模拟全命中但无 tokens
  GlobalCacheFetcher::PendingRequest pending;
  pending.req = req;
  pending.devices_completed = 0;
  pending.total_devices = 1;
  fetcher->pending_requests_[1500] = std::move(pending);

  GlobalCacheQueryResult r0;
  r0.req_id = 1500;
  r0.device_id = 0;
  r0.success = true;
  r0.matched_ranges = {0, 9};
  r0.has_cached_tokens = false;
  fetcher->OnDeviceComplete(r0);

  auto rejected = fetcher->TakeRejectedRequests();
  ASSERT_EQ(rejected.size(), 1u);
  EXPECT_TRUE(rejected[0]->finished);
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

// ==================== Decode 角色 UpdateRequestWorkload 成功路径 ====================
// 已有测试覆盖了 decode 角色的两种拒绝场景，此处补充全命中成功的路径。

class DecodeRoleUpdateWorkloadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // decode 角色，无 IPC（同步模式）
    fetcher_ = std::make_unique<GlobalCacheFetcher>(1, nullptr, "decode");
  }

  void CallUpdate(std::shared_ptr<InferRequest>& req, size_t new_cached_len, const std::vector<int>& cached_tokens = {},
                  bool has_cached_tokens = false) {
    fetcher_->UpdateRequestWorkload(req, new_cached_len, cached_tokens, has_cached_tokens);
  }

  std::unique_ptr<GlobalCacheFetcher> fetcher_;
};

TEST_F(DecodeRoleUpdateWorkloadTest, FullHitWithTokensSwitchesToDecode) {
  // 远端覆盖满 seq + tokens 就绪 → 切换 kDecode（短路，本地不算 Q）
  auto req = MakeFetcherTestRequest(2000, InferStage::kContext, 10);
  CallUpdate(req, /*new_cached_len=*/10, /*cached_tokens=*/{42, 43}, /*has_cached_tokens=*/true);

  EXPECT_FALSE(req->finished);
  EXPECT_EQ(req->infer_stage, InferStage::kDecode);
  auto planning = req->GetPlanningWorkload();
  EXPECT_EQ(planning.prefill_start_offset, 10u);
  EXPECT_EQ(planning.prefill_token_num, 2u);
}

// 无 tokens 的 seq-1 上限已由 UpdateWorkloadTest::FullHitWithoutTokensCapsToSeqMinusOne 覆盖；
// decode 角色无 tokens 的 reject 已由 FetcherDecodeRole::RejectsFullHitWithoutTokens 覆盖。

// ==================== SWA 对齐边界测试 ====================

TEST(SwaWindowBlockNumTest, DefaultsToOneBlockWhenWindowUnset) {
  GlobalCacheFetcher fetcher(1, nullptr);
  EXPECT_EQ(fetcher.SwaWindowBlockNum(16), 1u);
}

TEST(SwaWindowBlockNumTest, CeilDivideByBlockTokenNum) {
  GlobalCacheFetcher fetcher(1, nullptr);
  fetcher.SetSlidingWindowSize(4096);
  EXPECT_EQ(fetcher.SwaWindowBlockNum(16), 256u);

  fetcher.SetSlidingWindowSize(100);
  EXPECT_EQ(fetcher.SwaWindowBlockNum(16), 7u);
}

TEST(SwaWindowBlockNumTest, ZeroBlockTokenNumFallsBackToOne) {
  GlobalCacheFetcher fetcher(1, nullptr);
  fetcher.SetSlidingWindowSize(4096);
  EXPECT_EQ(fetcher.SwaWindowBlockNum(0), 1u);
}

// SWA 请求走两卡汇总，验证 L 一致 / 不一致两条路径。
class SwaFinalizeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fetcher_ = std::make_unique<GlobalCacheFetcher>(2, nullptr);
    fetcher_->SetSlidingWindowSize(16);
    fetcher_->Start();
  }

  void TearDown() override { fetcher_->Stop(); }

  // 造一个已分配 SWA block 的 SWA 请求，并按异步提交后的状态插入 pending。
  std::shared_ptr<InferRequest> InsertSwaPending(int64_t req_id, int token_num) {
    auto req = MakeFetcherTestRequest(req_id, InferStage::kContext, token_num);
    req->swa_kv_cache_blocks = {1, 2, 3};
    req->swa_kv_cache_block_idx_offsets = {0, 1, 2};

    GlobalCacheFetcher::PendingRequest pending;
    pending.req = req;
    pending.devices_completed = 0;
    pending.total_devices = 2;
    pending.swa_enabled = true;
    fetcher_->pending_requests_[req_id] = std::move(pending);
    return req;
  }

  void CompleteDevice(int64_t req_id, int device_id, const std::vector<int>& ranges, size_t swa_aligned_hit_len) {
    GlobalCacheQueryResult result;
    result.req_id = req_id;
    result.device_id = device_id;
    result.success = true;
    result.matched_ranges = ranges;
    result.swa_aligned_hit_len = swa_aligned_hit_len;
    fetcher_->OnDeviceComplete(result);
  }

  std::unique_ptr<GlobalCacheFetcher> fetcher_;
};

TEST_F(SwaFinalizeTest, ConsistentAlignedHitAdopted) {
  auto req = InsertSwaPending(3000, 48);
  CompleteDevice(3000, 0, {0, 31}, 32);
  CompleteDevice(3000, 1, {0, 31}, 32);

  ASSERT_EQ(fetcher_->TakeCompletedRequests().size(), 1u);
  EXPECT_EQ(req->GetPlanningWorkload().prefill_start_offset, 32u);
  // hook 需要按这个长度算 DropUnbackedSwaBlocks 的保留起点。
  EXPECT_EQ(req->global_swa_aligned_hit_len, 32u);
}

TEST_F(SwaFinalizeTest, InconsistentAlignedHitAbandoned) {
  auto req = InsertSwaPending(3100, 48);
  // 两卡对齐边界不同：取小值会让命中点落在缺 SWA 的卡上，只能整体放弃。
  CompleteDevice(3100, 0, {0, 31}, 32);
  CompleteDevice(3100, 1, {0, 31}, 16);

  ASSERT_EQ(fetcher_->TakeCompletedRequests().size(), 1u);
  EXPECT_EQ(req->GetPlanningWorkload().prefill_start_offset, 0u);
  EXPECT_EQ(req->global_swa_aligned_hit_len, 0u);
}

TEST_F(SwaFinalizeTest, ZeroAlignedHitFromOneDeviceAbandoned) {
  auto req = InsertSwaPending(3200, 48);
  CompleteDevice(3200, 0, {0, 31}, 32);
  CompleteDevice(3200, 1, {0, 31}, 0);

  ASSERT_EQ(fetcher_->TakeCompletedRequests().size(), 1u);
  EXPECT_EQ(req->GetPlanningWorkload().prefill_start_offset, 0u);
  EXPECT_EQ(req->global_swa_aligned_hit_len, 0u);
}

TEST_F(SwaFinalizeTest, AlignedHitMismatchesContiguousPrefixAbandoned) {
  auto req = InsertSwaPending(3300, 48);
  // 主 KV 连续前缀只有 16，但报回的对齐边界是 32：两个信号对不上即放弃。
  CompleteDevice(3300, 0, {0, 15}, 32);
  CompleteDevice(3300, 1, {0, 15}, 32);

  ASSERT_EQ(fetcher_->TakeCompletedRequests().size(), 1u);
  EXPECT_EQ(req->GetPlanningWorkload().prefill_start_offset, 0u);
  EXPECT_EQ(req->global_swa_aligned_hit_len, 0u);
}

TEST_F(UpdateWorkloadTest, SwaAlignedHitTruncatedByCapAbandoned) {
  // 对齐边界等于 seq_len 时会被 seq-1 上限改写，说明边界与「有 SWA 的位置」不再对应，放弃全局命中。
  auto req = MakeFetcherTestRequest(3400, InferStage::kContext, 32);
  req->swa_kv_cache_blocks = {1, 2};
  req->block_token_num = 16;

  fetcher_->UpdateRequestWorkload(req, /*new_cached_len=*/32, /*cached_tokens=*/{}, /*has_cached_tokens=*/false,
                                  /*swa_aligned_hit_len=*/32);

  auto planning = req->GetPlanningWorkload();
  EXPECT_EQ(planning.prefill_start_offset, 0u);
  EXPECT_EQ(planning.prefill_token_num, 32u);
}

TEST_F(UpdateWorkloadTest, SwaAlignedHitWithinCapApplied) {
  auto req = MakeFetcherTestRequest(3500, InferStage::kContext, 48);
  req->swa_kv_cache_blocks = {1, 2, 3};
  req->block_token_num = 16;

  fetcher_->UpdateRequestWorkload(req, /*new_cached_len=*/32, /*cached_tokens=*/{}, /*has_cached_tokens=*/false,
                                  /*swa_aligned_hit_len=*/32);

  auto planning = req->GetPlanningWorkload();
  EXPECT_EQ(planning.prefill_start_offset, 32u);
  EXPECT_EQ(planning.prefill_token_num, 16u);
}

// ==================== SweepTimeouts / DispatchQueryToExecutors ====================

TEST(SweepTimeoutsTest, ForcesFinalizeWhenAgedOut) {
  auto fetcher = std::make_unique<GlobalCacheFetcher>(/*device_count=*/2, /*ipc_func_wrapper=*/nullptr, "",
                                                      /*fetch_thread_num=*/1, /*fetch_timeout_ms=*/1);
  auto req = MakeFetcherTestRequest(4000, InferStage::kContext, 16);

  GlobalCacheFetcher::PendingRequest pending;
  pending.req = req;
  pending.total_devices = 2;
  pending.devices_completed = 0;
  pending.device_done_us.assign(2, -1);
  pending.submit_time_us = ProfileTimer::GetCurrentTimeInUs() - 10'000;
  fetcher->pending_requests_[4000] = std::move(pending);

  fetcher->SweepTimeouts();

  ASSERT_EQ(fetcher->TakeCompletedRequests().size(), 1u);
  EXPECT_FALSE(fetcher->HasPendingRequests());
}

TEST(SweepTimeoutsTest, SkipsWhenStillWithinTimeout) {
  auto fetcher = std::make_unique<GlobalCacheFetcher>(/*device_count=*/1, /*ipc_func_wrapper=*/nullptr, "",
                                                      /*fetch_thread_num=*/1, /*fetch_timeout_ms=*/5000);
  auto req = MakeFetcherTestRequest(4100, InferStage::kContext, 16);

  GlobalCacheFetcher::PendingRequest pending;
  pending.req = req;
  pending.total_devices = 1;
  pending.devices_completed = 0;
  pending.device_done_us.assign(1, -1);
  pending.submit_time_us = ProfileTimer::GetCurrentTimeInUs();
  fetcher->pending_requests_[4100] = std::move(pending);

  fetcher->SweepTimeouts();

  EXPECT_TRUE(fetcher->HasPendingRequests());
  EXPECT_TRUE(fetcher->TakeCompletedRequests().empty());
  fetcher->pending_requests_.clear();
}

TEST(SweepTimeoutsTest, PartialDeviceDoneThenTimeout) {
  auto fetcher = std::make_unique<GlobalCacheFetcher>(/*device_count=*/2, /*ipc_func_wrapper=*/nullptr, "",
                                                      /*fetch_thread_num=*/1, /*fetch_timeout_ms=*/1);
  auto req = MakeFetcherTestRequest(4200, InferStage::kContext, 16);

  GlobalCacheFetcher::PendingRequest pending;
  pending.req = req;
  pending.total_devices = 2;
  pending.devices_completed = 1;
  pending.device_done_us = {100, -1};
  pending.per_device_ranges.push_back({0, 9});
  pending.submit_time_us = ProfileTimer::GetCurrentTimeInUs() - 10'000;
  fetcher->pending_requests_[4200] = std::move(pending);

  fetcher->SweepTimeouts();

  // 超时强制失败收尾，不再扩展 global prefix。
  ASSERT_EQ(fetcher->TakeCompletedRequests().size(), 1u);
  EXPECT_EQ(req->GetPlanningWorkload().prefill_start_offset, 0u);
}

TEST(DispatchQueryToExecutorsTest, InvalidExecutorIdMarksDeviceFailed) {
  auto wrapper = std::make_shared<IpcFuncWrapper>(/*executor_num=*/1, /*is_engine=*/true, /*executor_id=*/0);
  auto fetcher = std::make_unique<GlobalCacheFetcher>(/*device_count=*/2, wrapper, "", /*fetch_thread_num=*/1,
                                                      /*fetch_timeout_ms=*/1);

  auto req = MakeFetcherTestRequest(4300, InferStage::kContext, 32);
  req->kv_cache_blocks = {10, 11};
  req->block_token_num = 16;
  req->swa_kv_cache_blocks = {20, 21};
  req->swa_kv_cache_block_idx_offsets = {0, 1};
  fetcher->SetSlidingWindowSize(16);

  GlobalCacheFetcher::PendingRequest pending;
  pending.req = req;
  pending.total_devices = 2;
  pending.devices_completed = 0;
  pending.device_done_us.assign(2, -1);
  pending.swa_enabled = true;
  pending.query_seq_len = 32;
  pending.submit_time_us = ProfileTimer::GetCurrentTimeInUs() - 10'000;
  fetcher->pending_requests_[4300] = std::move(pending);

  fetcher->DispatchQueryToExecutors(fetcher->pending_requests_[4300]);
  fetcher->SweepTimeouts();

  ASSERT_EQ(fetcher->TakeCompletedRequests().size(), 1u);
  EXPECT_FALSE(fetcher->HasPendingRequests());

  wrapper->GetIpcFuncProxy()->UnlinkIpcResource();
}

}  // namespace
}  // namespace ksana_llm
