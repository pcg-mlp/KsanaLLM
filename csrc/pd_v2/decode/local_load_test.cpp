/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/decode/local_load.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/event_driven_state.h"
#include "batch_scheduler/event_driven_scheduler/schedule_context.h"
#include "cache_manager/cache_manager_interface.h"
#include "runtime/infer_request.h"
#include "utils/request.h"

namespace ksana_llm {
namespace pd_v2 {
namespace {

// Minimal CacheManagerInterface mock — only GetUsableBlockNumber is
// exercised by ReadLocalLoad; the rest are stubbed so the abstract base
// can be instantiated.
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
              (int64_t, size_t, std::vector<int>&, std::vector<int>&, std::vector<size_t>&), (override));
  MOCK_METHOD(void, UpdateFlexibleCache,
              (int64_t, const std::vector<int>&, int, std::vector<FlexibleCachedCopyTask>&, size_t&), (override));
  MOCK_METHOD(Status, UpdateRequestTokens,
              (int64_t, const std::vector<int>&, size_t, std::vector<int>&, bool&, std::vector<int>&,
               std::vector<size_t>&),
              (override));
  MOCK_METHOD(void, DestroyFinishedRequest, (int64_t), (override));
  MOCK_METHOD(bool, IsPrefixCachingEnabled, (), (override));
};

// Keep Request objects alive: InferRequest holds a reference to its Request.
std::vector<std::shared_ptr<Request>> g_alive_requests;

// Build an InferRequest whose GetPlanningSequenceLen() == planning_len. A
// fresh ScheduleTaskWorkload is all-zero, so the planning length reduces to
// forwarding_tokens.size().
std::shared_ptr<InferRequest> MakeReq(int64_t req_id, size_t planning_len) {
  auto input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, req_ctx);
  g_alive_requests.push_back(request);
  auto infer_req = std::make_shared<InferRequest>(request, 0);
  infer_req->req_id = req_id;
  infer_req->forwarding_tokens.assign(planning_len, 1);
  return infer_req;
}

}  // namespace

TEST(LocalLoadTest, CountsQueuesAndSumsPendingTokens) {
  ScheduleState schedule_state;
  schedule_state.schedule_states.resize(1);
  schedule_state.schedule_states[0].resize(1);
  DpScheduleState& dp = schedule_state.schedule_states[0][0];
  dp.prefilling_reqs.push_back(MakeReq(1, 10));
  dp.prefilling_reqs.push_back(MakeReq(2, 25));
  dp.prefilling_reqs.push_back(MakeReq(3, 7));
  dp.decoding_reqs.push_back(MakeReq(4, 1));
  dp.decoding_reqs.push_back(MakeReq(5, 1));

  auto mock_cm = std::make_shared<testing::NiceMock<MockCacheManager>>();
  ON_CALL(*mock_cm, GetUsableBlockNumber()).WillByDefault(testing::Return(4096));
  std::vector<std::shared_ptr<CacheManagerInterface>> cms{mock_cm};

  ScheduleContext ctx;
  ctx.SetScheduleState(&schedule_state);
  ctx.SetDpCacheManagers(&cms);

  LocalLoadView view = ReadLocalLoad(ctx, 0, /*step_token_num=*/2048, /*step_latency_us=*/31000);

  EXPECT_EQ(view.waiting_reqs, 3u);
  EXPECT_EQ(view.running_reqs, 2u);
  EXPECT_EQ(view.pending_prefill_tokens, 42u);  // 10 + 25 + 7
  EXPECT_EQ(view.free_kv_blocks, 4096u);
  // swap removed: future-free is always 0.
  EXPECT_EQ(view.future_free_kv_blocks, 0u);
  EXPECT_EQ(view.step_token_num, 2048u);
  EXPECT_EQ(view.step_latency_us, 31000u);
}

TEST(LocalLoadTest, NullCacheManagerLeavesBlockFieldsZero) {
  ScheduleState schedule_state;
  schedule_state.schedule_states.resize(1);
  schedule_state.schedule_states[0].resize(1);
  schedule_state.schedule_states[0][0].prefilling_reqs.push_back(MakeReq(1, 5));

  std::vector<std::shared_ptr<CacheManagerInterface>> cms{nullptr};
  ScheduleContext ctx;
  ctx.SetScheduleState(&schedule_state);
  ctx.SetDpCacheManagers(&cms);

  LocalLoadView view = ReadLocalLoad(ctx, 0);

  EXPECT_EQ(view.waiting_reqs, 1u);
  EXPECT_EQ(view.running_reqs, 0u);
  EXPECT_EQ(view.pending_prefill_tokens, 5u);
  EXPECT_EQ(view.free_kv_blocks, 0u);
  EXPECT_EQ(view.future_free_kv_blocks, 0u);
  EXPECT_EQ(view.step_token_num, 0u);
  EXPECT_EQ(view.step_latency_us, 0u);
}

TEST(LocalLoadTest, SkipsNullRequestsInPendingSum) {
  ScheduleState schedule_state;
  schedule_state.schedule_states.resize(1);
  schedule_state.schedule_states[0].resize(1);
  DpScheduleState& dp = schedule_state.schedule_states[0][0];
  dp.prefilling_reqs.push_back(MakeReq(1, 12));
  dp.prefilling_reqs.push_back(nullptr);  // must not be dereferenced
  dp.prefilling_reqs.push_back(MakeReq(2, 8));

  std::vector<std::shared_ptr<CacheManagerInterface>> cms{nullptr};
  ScheduleContext ctx;
  ctx.SetScheduleState(&schedule_state);
  ctx.SetDpCacheManagers(&cms);

  LocalLoadView view = ReadLocalLoad(ctx, 0);

  EXPECT_EQ(view.waiting_reqs, 3u);  // size counts the null slot too
  EXPECT_EQ(view.pending_prefill_tokens, 20u);
}

}  // namespace pd_v2
}  // namespace ksana_llm
