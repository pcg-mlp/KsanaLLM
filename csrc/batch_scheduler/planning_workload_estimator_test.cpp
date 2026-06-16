/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/planning_workload_estimator.h"

#include <gtest/gtest.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "runtime/infer_request.h"
#include "utils/request.h"

namespace ksana_llm {

class PlanningWorkloadEstimatorTest : public testing::Test {
 protected:
  void SetUp() override { InitLoguru(); }

  // Helper to create an InferRequest via Request (which requires KsanaPythonInput).
  std::shared_ptr<InferRequest> CreateInferRequest(const std::vector<int>& input_tokens) {
    auto python_input = std::make_shared<KsanaPythonInput>();
    python_input->input_tokens = input_tokens;

    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    request_ = std::make_shared<Request>(python_input, req_ctx);
    python_input_ = python_input;  // prevent KsanaPythonInput from being destroyed

    return std::make_shared<InferRequest>(request_, 0);
  }

  // Set up an inflight prefill task for the infer request.
  // prefill_token_num: number of tokens to prefill in this step
  // prefill_start_offset: starting offset for this prefill chunk
  void SetupInflightPrefillTask(std::shared_ptr<InferRequest>& req, size_t prefill_token_num,
                                size_t prefill_start_offset) {
    ScheduleTaskWorkload workload;
    workload.prefill_token_num = prefill_token_num;
    workload.prefill_start_offset = prefill_start_offset;
    workload.sampling_token_num = 1;
    req->SetPlanningWorkload(workload);
    req->SetPlanningTask();
    req->LaunchPlanningTask();
  }

  std::shared_ptr<KsanaPythonInput> python_input_;
  std::shared_ptr<Request> request_;
};

// Test: EstimateRequestPlanningWorkload sets zero estimation for intermediate split-fuse prefill steps.
TEST_F(PlanningWorkloadEstimatorTest, IntermediatePrefillSetsZero) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5, 6});
  req->output_tokens = {1, 2, 3, 4, 5, 6};
  req->ResetPrefillingTokens();

  // Intermediate prefill step: 3 out of 6 tokens.
  SetupInflightPrefillTask(req, /*prefill_token_num=*/3, /*prefill_start_offset=*/0);

  const size_t max_draft = 4;
  EstimateRequestPlanningWorkload(req, max_draft);

  // For intermediate prefill steps, estimation should be (0, 0).
  const auto& workload = req->GetPlanningWorkload();
  EXPECT_EQ(workload.generated_token_num, 0);
  EXPECT_EQ(workload.draft_token_num, 0);
}

// Test: EstimateRequestPlanningWorkload sets full estimation for last prefill step.
TEST_F(PlanningWorkloadEstimatorTest, LastPrefillSetsFull) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5, 6});
  req->output_tokens = {1, 2, 3, 4, 5, 6};
  req->ResetPrefillingTokens();

  // First chunk done.
  SetupInflightPrefillTask(req, /*prefill_token_num=*/3, /*prefill_start_offset=*/0);
  req->UpdateAfterInflightTaskFinished();
  req->ResetInflightTask();

  // Last prefill step: offset=3, num=3 → 3+3 == 6.
  SetupInflightPrefillTask(req, /*prefill_token_num=*/3, /*prefill_start_offset=*/3);

  const size_t max_draft = 4;
  EstimateRequestPlanningWorkload(req, max_draft);

  // For last prefill step, estimation should use kStepGenerateTokenNum and max_draft.
  const auto& workload = req->GetPlanningWorkload();
  EXPECT_EQ(workload.generated_token_num, kStepGenerateTokenNum);
  EXPECT_EQ(workload.draft_token_num, max_draft);
}

// build-ahead 第二步 schedule 时已有 decode inflight：应按 kStepGenerateTokenNum 估算 generated/draft。
TEST_F(PlanningWorkloadEstimatorTest, DecodeInflightSetsWorkloadEstimation) {
  auto req = CreateInferRequest({1, 2, 3});
  req->output_tokens = {1, 2, 3};
  req->ResetPrefillingTokens();
  // 完成 prefill
  SetupInflightPrefillTask(req, /*prefill_token_num=*/3, /*prefill_start_offset=*/0);
  req->generated_tokens = {10};
  req->UpdateAfterInflightTaskFinished();
  req->ResetInflightTask();

  // 第一步 decode 已 launch，尚未 retire
  ScheduleTaskWorkload decode_wl;
  decode_wl.generated_token_num = 1;
  decode_wl.draft_token_num = 0;
  decode_wl.sampling_token_num = 1;
  req->SetPlanningWorkload(decode_wl);
  req->SetPlanningTask();
  req->LaunchPlanningTask();
  ASSERT_TRUE(req->HasInflightTask());

  const size_t max_draft = 4;
  EstimateRequestPlanningWorkload(req, max_draft);

  const auto& planning = req->GetPlanningWorkload();
  EXPECT_EQ(planning.generated_token_num, kStepGenerateTokenNum);
  EXPECT_EQ(planning.draft_token_num, max_draft);
}

// Test: EstimateRequestPlanningWorkload with no inflight task just copies remaining workload.
TEST_F(PlanningWorkloadEstimatorTest, NoInflightTask) {
  auto req = CreateInferRequest({1, 2, 3});
  req->output_tokens = {1, 2, 3};
  req->ResetPrefillingTokens();

  // Set a remaining workload manually without launching a task.
  ScheduleTaskWorkload workload;
  workload.prefill_token_num = 3;
  workload.generated_token_num = 0;
  workload.draft_token_num = 0;
  req->SetRemainingWorkload(workload);

  // No inflight task → HasInflightTask() is false.
  EXPECT_FALSE(req->HasInflightTask());

  EstimateRequestPlanningWorkload(req, /*max_draft_token_num=*/2);

  // Planning workload should simply equal the remaining workload.
  const auto& planning = req->GetPlanningWorkload();
  EXPECT_EQ(planning.prefill_token_num, 3);
  EXPECT_EQ(planning.generated_token_num, 0);
  EXPECT_EQ(planning.draft_token_num, 0);
}

}  // namespace ksana_llm
