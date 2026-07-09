/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================
 */

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/event_driven_state.h"
#include "batch_scheduler/event_driven_scheduler/schedule_context.h"
#include "batch_scheduler/event_driven_scheduler/workload_balance/basic_workload_balancer.h"
#include "runtime/infer_request.h"
#include "runtime/infer_stage.h"
#include "tests/test.h"
#include "utils/request.h"

namespace ksana_llm {
namespace {

class BasicWorkloadBalancerTest : public testing::Test {
 protected:
  std::shared_ptr<InferRequest> CreateInferRequest(int64_t req_id, size_t planning_query_len,
                                                   size_t sampling_token_num) {
    auto input = std::make_shared<KsanaPythonInput>();
    input->input_tokens.assign(planning_query_len, 1);
    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    auto request = std::make_shared<Request>(input, req_ctx);
    auto infer_req = std::make_shared<InferRequest>(request, 0);
    infer_req->req_id = req_id;
    infer_req->sampling_token_num = sampling_token_num;
    infer_req->output_tokens.resize(planning_query_len);  // Prepare for prefilling_tokens.
    infer_req->ResetPrefillingTokens();

    ScheduleTaskWorkload planning_workload;
    planning_workload.prefill_token_num = planning_query_len;
    planning_workload.sampling_token_num = sampling_token_num;
    infer_req->SetRemainingWorkload(planning_workload);
    infer_req->SetPlanningWorkload(planning_workload);

    input_holders_.push_back(std::move(input));
    request_holders_.push_back(std::move(request));
    return infer_req;
  }

 private:
  std::vector<std::shared_ptr<KsanaPythonInput>> input_holders_;
  std::vector<std::shared_ptr<Request>> request_holders_;
};

TEST_F(BasicWorkloadBalancerTest, SingleDpBatchAssembleRespectsBatchLimits) {
  BasicWorkloadBalancer balancer(/*dp_num=*/1, /*max_batch_size=*/2, /*max_decode_tokens_per_req=*/4,
                                 /*max_step_token_num=*/10);

  ScheduleState schedule_state;
  schedule_state.schedule_states.resize(1);
  schedule_state.schedule_states[0].resize(1);
  ScheduleContext context;
  context.SetScheduleState(&schedule_state);

  std::vector<std::shared_ptr<InferRequest>> requests;
  requests.push_back(CreateInferRequest(/*req_id=*/1001, /*planning_query_len=*/4, /*sampling_token_num=*/1));
  requests.push_back(CreateInferRequest(/*req_id=*/1002, /*planning_query_len=*/3, /*sampling_token_num=*/2));
  requests.push_back(CreateInferRequest(/*req_id=*/1003, /*planning_query_len=*/4, /*sampling_token_num=*/2));

  balancer.AddInferRequests(requests, context);
  for (const auto& req : requests) {
    ASSERT_NE(req, nullptr);
    EXPECT_EQ(req->attn_dp_group_id, 0u);
  }

  PlanningBatch batch;
  batch.dp_batches.resize(1);
  WorkloadQuota quota{0, 10};

  ASSERT_TRUE(balancer.IsRequestAcceptable(requests[0], batch, quota, /*seq_token_num=*/4,
                                           /*per_dp_step_token_num=*/4));
  batch.AddRunningReq(0, requests[0]);
  ASSERT_TRUE(balancer.IsRequestAcceptable(requests[1], batch, quota, /*seq_token_num=*/3,
                                           /*per_dp_step_token_num=*/3));
  batch.AddRunningReq(0, requests[1]);

  // The 3rd request should be rejected because max_batch_size is reached.
  EXPECT_FALSE(balancer.IsRequestAcceptable(requests[2], batch, quota, /*seq_token_num=*/4,
                                            /*per_dp_step_token_num=*/4));
}

TEST_F(BasicWorkloadBalancerTest, MultiDpAssignAndQuotaFollowLoadBalance) {
  BasicWorkloadBalancer balancer(/*dp_num=*/3, /*max_batch_size=*/8, /*max_decode_tokens_per_req=*/4,
                                 /*max_step_token_num=*/12);

  ScheduleState schedule_state;
  schedule_state.schedule_states.resize(1);
  schedule_state.schedule_states[0].resize(3);
  schedule_state.schedule_states[0][0].decoding_reqs.push_back(CreateInferRequest(1, 1, 1));
  schedule_state.schedule_states[0][0].decoding_reqs.push_back(CreateInferRequest(2, 1, 1));
  schedule_state.schedule_states[0][2].decoding_reqs.push_back(CreateInferRequest(3, 1, 1));

  ScheduleContext context;
  context.SetScheduleState(&schedule_state);

  std::vector<std::shared_ptr<InferRequest>> incoming;
  incoming.push_back(CreateInferRequest(100, 2, 1));
  incoming.push_back(CreateInferRequest(101, 2, 1));
  incoming.push_back(CreateInferRequest(102, 2, 1));
  incoming.push_back(CreateInferRequest(103, 2, 1));

  balancer.AddInferRequests(incoming, context);
  ASSERT_EQ(incoming.size(), 4u);
  EXPECT_EQ(incoming[0]->attn_dp_group_id, 1u);
  EXPECT_EQ(incoming[1]->attn_dp_group_id, 2u);
  EXPECT_EQ(incoming[2]->attn_dp_group_id, 1u);
  EXPECT_EQ(incoming[3]->attn_dp_group_id, 2u);

  PlanningBatch batch;
  batch.dp_batches.resize(3);
  batch.dp_batches[0].running_step_token_num = 6;
  batch.dp_batches[1].running_step_token_num = 2;
  batch.dp_batches[2].running_step_token_num = 4;

  const auto quotas = balancer.GetDpSchedulingOrderAndQuota(batch, context, InferStage::kDecode);
  ASSERT_EQ(quotas.size(), 6u);

  // small_quota = max_step_token_num / (dp_num * 2) = 12 / 6 = 2
  EXPECT_EQ(quotas[0].dp_idx, 1u);
  EXPECT_EQ(quotas[1].dp_idx, 2u);
  EXPECT_EQ(quotas[2].dp_idx, 0u);
  EXPECT_EQ(quotas[0].step_token_limit, 2);
  EXPECT_EQ(quotas[1].step_token_limit, 2);
  EXPECT_EQ(quotas[2].step_token_limit, 2);

  // large_quota = max_step_token_num / dp_num = 12 / 3 = 4
  EXPECT_EQ(quotas[3].dp_idx, 1u);
  EXPECT_EQ(quotas[4].dp_idx, 2u);
  EXPECT_EQ(quotas[5].dp_idx, 0u);
  EXPECT_EQ(quotas[3].step_token_limit, 4);
  EXPECT_EQ(quotas[4].step_token_limit, 4);
  EXPECT_EQ(quotas[5].step_token_limit, 4);
}

TEST_F(BasicWorkloadBalancerTest, MultiDpRejectWhenGlobalPlanningBatchQuotaExceeded) {
  BasicWorkloadBalancer balancer(/*dp_num=*/2, /*max_batch_size=*/8, /*max_decode_tokens_per_req=*/4,
                                 /*max_step_token_num=*/10);

  auto req = CreateInferRequest(/*req_id=*/2001, /*planning_query_len=*/3, /*sampling_token_num=*/1);
  ASSERT_NE(req, nullptr);

  WorkloadQuota quota{/*dp_idx=*/1, /*step_token_limit=*/5};

  auto req_dp1_ok = CreateInferRequest(/*req_id=*/2002, /*planning_query_len=*/1, /*sampling_token_num=*/1);
  auto req_dp1_reject = CreateInferRequest(/*req_id=*/2005, /*planning_query_len=*/1, /*sampling_token_num=*/1);
  auto req_dp0_ok = CreateInferRequest(/*req_id=*/2003, /*planning_query_len=*/6, /*sampling_token_num=*/1);
  auto req_dp0_reject = CreateInferRequest(/*req_id=*/2004, /*planning_query_len=*/7, /*sampling_token_num=*/1);
  ASSERT_NE(req_dp1_ok, nullptr);
  ASSERT_NE(req_dp1_reject, nullptr);
  ASSERT_NE(req_dp0_ok, nullptr);
  ASSERT_NE(req_dp0_reject, nullptr);

  PlanningBatch batch_ok;
  batch_ok.dp_batches.resize(2);
  batch_ok.AddRunningReq(/*dp_idx=*/1, req_dp1_ok);  // dp1 running_step_token_num = 1
  batch_ok.AddRunningReq(/*dp_idx=*/0, req_dp0_ok);  // global running_step_token_num = 7

  // Within dp quota: 1 + 3 <= 5, and within global planning batch quota: 7 + 3 <= 10.
  EXPECT_TRUE(balancer.IsRequestAcceptable(req, batch_ok, quota, /*seq_token_num=*/3,
                                           /*per_dp_step_token_num=*/3));

  PlanningBatch batch_reject;
  batch_reject.dp_batches.resize(2);
  batch_reject.AddRunningReq(/*dp_idx=*/1, req_dp1_reject);  // dp1 running_step_token_num = 1
  batch_reject.AddRunningReq(/*dp_idx=*/0, req_dp0_reject);  // global running_step_token_num = 8

  // Still within dp quota, but exceeds global planning batch quota: 8 + 3 > 10.
  EXPECT_FALSE(balancer.IsRequestAcceptable(req, batch_reject, quota, /*seq_token_num=*/3,
                                            /*per_dp_step_token_num=*/3));
}

TEST_F(BasicWorkloadBalancerTest, MultiDpRejectWhenGlobalRunningReqNumReachedMaxBatchSize) {
  BasicWorkloadBalancer balancer(/*dp_num=*/2, /*max_batch_size=*/2, /*max_decode_tokens_per_req=*/4,
                                 /*max_step_token_num=*/10);

  auto req_dp0 = CreateInferRequest(/*req_id=*/3001, /*planning_query_len=*/2, /*sampling_token_num=*/1);
  auto req_dp1 = CreateInferRequest(/*req_id=*/3002, /*planning_query_len=*/2, /*sampling_token_num=*/1);
  auto req_new = CreateInferRequest(/*req_id=*/3003, /*planning_query_len=*/2, /*sampling_token_num=*/1);
  ASSERT_NE(req_dp0, nullptr);
  ASSERT_NE(req_dp1, nullptr);
  ASSERT_NE(req_new, nullptr);

  PlanningBatch batch;
  batch.dp_batches.resize(2);
  batch.AddRunningReq(/*dp_idx=*/0, req_dp0);
  batch.AddRunningReq(/*dp_idx=*/1, req_dp1);

  // total running reqs across all dp batches is already 2, equal to max_batch_size.
  WorkloadQuota quota{/*dp_idx=*/0, /*step_token_limit=*/10};
  EXPECT_FALSE(balancer.IsRequestAcceptable(req_new, batch, quota, /*seq_token_num=*/2,
                                            /*per_dp_step_token_num=*/2));
}

// 单请求豁免: per-dp quota 只在 dp 本步已有请求时限制(防多请求竞争吃光预算)。dp 空时单个大请求
// (seq > per-dp quota)应被豁免, 只受全局 max_step_token_num 约束 —— 否则 seq > max_step/dp 的
// 请求被死拒, 在 EDS 下触发 recompute 活锁(standalone 永久死锁)。
TEST_F(BasicWorkloadBalancerTest, SingleRequestExemptFromPerDpQuotaWhenDpEmpty) {
  BasicWorkloadBalancer balancer(/*dp_num=*/8, /*max_batch_size=*/8, /*max_decode_tokens_per_req=*/4,
                                 /*max_step_token_num=*/65536);
  auto big_req = CreateInferRequest(/*req_id=*/4001, /*planning_query_len=*/9000, /*sampling_token_num=*/1);
  ASSERT_NE(big_req, nullptr);

  // per-dp large_quota = max_step_token_num / dp_num = 65536 / 8 = 8192 < 9000。
  WorkloadQuota quota{/*dp_idx=*/0, /*step_token_limit=*/8192};

  // dp 空: 单请求成批 → 豁免 per-dp quota, 只受全局 65536 约束, 9000 可接受。
  PlanningBatch batch_empty;
  batch_empty.dp_batches.resize(8);
  EXPECT_TRUE(balancer.IsRequestAcceptable(big_req, batch_empty, quota, /*seq_token_num=*/9000,
                                           /*per_dp_step_token_num=*/9000));

  // dp 非空: per-dp quota 生效, dp0 已有 1 token, 1 + 9000 > 8192 → 拒绝(避免挤占同 dp 其它请求)。
  PlanningBatch batch_busy;
  batch_busy.dp_batches.resize(8);
  auto occupant = CreateInferRequest(/*req_id=*/4002, /*planning_query_len=*/1, /*sampling_token_num=*/1);
  batch_busy.AddRunningReq(/*dp_idx=*/0, occupant);
  EXPECT_FALSE(balancer.IsRequestAcceptable(big_req, batch_busy, quota, /*seq_token_num=*/9000,
                                            /*per_dp_step_token_num=*/9000));
}

}  // namespace
}  // namespace ksana_llm
