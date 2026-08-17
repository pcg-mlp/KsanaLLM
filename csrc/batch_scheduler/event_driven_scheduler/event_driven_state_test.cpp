/* Copyright 2026 Tencent Inc. All rights reserved.

==============================================================================*/

// PlanningBatch::IsLaunchable(max_depth) 单测，语义与 ScheduleOutput::IsLaunchable 对齐。

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/event_driven_state.h"
#include "runtime/infer_request.h"
#include "runtime/infer_stage.h"
#include "utils/request.h"

namespace ksana_llm {

namespace {

struct DecodeReadyInferRequestFixture {
  std::shared_ptr<Request> request;
  std::shared_ptr<InferRequest> infer_req;
};

DecodeReadyInferRequestFixture BuildDecodeReadyInferRequest(int64_t req_id, int first_decoded_token = 10) {
  auto python_input = std::make_shared<KsanaPythonInput>();
  python_input->input_tokens = {1, 2, 3};
  auto ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(python_input, ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->req_id = req_id;
  req->output_tokens = {1, 2, 3};
  req->ResetPrefillingTokens();

  ScheduleTaskWorkload prefill_wl;
  prefill_wl.prefill_token_num = 3;
  prefill_wl.prefill_start_offset = 0;
  prefill_wl.sampling_token_num = 1;
  req->SetPlanningWorkload(prefill_wl);
  req->SetPlanningTask();
  req->LaunchPlanningTask();
  req->generated_tokens = {first_decoded_token};
  req->UpdateAfterInflightTaskFinished();
  req->ResetInflightTask();
  return {request, req};
}

void ArmDecodePlanning(const std::shared_ptr<InferRequest>& req) {
  ScheduleTaskWorkload wl;
  wl.generated_token_num = 1;
  wl.draft_token_num = 0;
  wl.sampling_token_num = 1;
  req->SetPlanningWorkload(wl);
}

// Prefill step that hits the prefix cache: prefix_len tokens are already in KV and only chunk_len
// tokens are forwarded. Mirrors the request state ProcessPrefillingQueue leaves before
// AddRunningReq (prefill_start_offset already advanced to the shared token count).
DecodeReadyInferRequestFixture BuildPrefixHitPrefillRequest(int64_t req_id, size_t prefix_len, size_t chunk_len) {
  auto python_input = std::make_shared<KsanaPythonInput>();
  python_input->input_tokens.assign(prefix_len + chunk_len, 1);
  auto ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(python_input, ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->req_id = req_id;
  req->output_tokens.assign(prefix_len + chunk_len, 1);
  req->ResetPrefillingTokens();

  ScheduleTaskWorkload wl;
  wl.prefill_token_num = chunk_len;
  wl.prefill_start_offset = prefix_len;
  wl.sampling_token_num = 1;
  req->SetRemainingWorkload(wl);
  req->SetPlanningWorkload(wl);
  return {request, req};
}

// Request whose last prefill chunk is still inflight: kMoveToDecode has already moved it to the
// decoding queue, but infer_stage only flips to kDecode when the generation result returns.
DecodeReadyInferRequestFixture BuildInflightPrefillInferRequest(int64_t req_id) {
  auto python_input = std::make_shared<KsanaPythonInput>();
  python_input->input_tokens = {1, 2, 3};
  auto ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(python_input, ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->req_id = req_id;
  req->output_tokens = {1, 2, 3};
  req->ResetPrefillingTokens();

  ScheduleTaskWorkload prefill_wl;
  prefill_wl.prefill_token_num = 3;
  prefill_wl.prefill_start_offset = 0;
  prefill_wl.sampling_token_num = 1;
  req->SetPlanningWorkload(prefill_wl);
  req->SetPlanningTask();
  req->LaunchPlanningTask();
  return {request, req};
}

}  // namespace

class EventDrivenStateTest : public testing::Test {
 protected:
  void SetUp() override { InitLoguru(); }
};

TEST_F(EventDrivenStateTest, PlanningBatchIsLaunchableRespectsMaxDepth) {
  auto fixture = BuildDecodeReadyInferRequest(/*req_id=*/3001);
  auto& req = fixture.infer_req;
  PlanningBatch batch;
  batch.dp_batches.resize(1);
  batch.running_req_num = 1;
  batch.dp_batches[0].running_reqs.emplace(req->req_id, req);

  EXPECT_TRUE(batch.IsLaunchable(/*max_depth=*/1));
  EXPECT_TRUE(batch.IsLaunchable(/*max_depth=*/2));

  ArmDecodePlanning(req);
  req->SetPlanningTask();
  req->LaunchPlanningTask();
  ASSERT_EQ(req->GetInflightCount(), 1u);

  EXPECT_FALSE(batch.IsLaunchable(/*max_depth=*/1));
  EXPECT_TRUE(batch.IsLaunchable(/*max_depth=*/2));

  req->generated_tokens.clear();
  ArmDecodePlanning(req);
  req->SetPlanningTask();
  req->LaunchPlanningTask(/*use_placeholder_for_last_token=*/true);
  ASSERT_EQ(req->GetInflightCount(), 2u);
  EXPECT_FALSE(batch.IsLaunchable(/*max_depth=*/2));
}

// A prefix-cache hit (and, identically, the chunks a split-fuse request already computed) is
// admitted on the uncached chunk, so it must be charged on the chunk too. Charging
// prefill_start_offset + chunk instead blows running_step_token_num past max_step_token_num and
// blocks every later request in the step.
TEST_F(EventDrivenStateTest, AddRunningReqChargesPrefillChunkNotPrefixHit) {
  auto fixture = BuildPrefixHitPrefillRequest(/*req_id=*/3005, /*prefix_len=*/16, /*chunk_len=*/4);
  PlanningBatch batch;
  batch.dp_batches.resize(1);

  batch.AddRunningReq(/*dp_idx=*/0, fixture.infer_req);

  EXPECT_EQ(batch.running_step_token_num, 4u);
  EXPECT_EQ(batch.dp_batches[0].running_step_token_num, 4u);

  batch.RemoveRunningReq(/*dp_idx=*/0, fixture.infer_req->req_id);
  EXPECT_EQ(batch.running_step_token_num, 0u);
  EXPECT_EQ(batch.dp_batches[0].running_step_token_num, 0u);
}

// While the last prefill chunk is inflight the request is planned as decode but infer_stage is
// still kContext. Only the decode query may be charged, never forwarding_tokens.size().
TEST_F(EventDrivenStateTest, AddRunningReqChargesDecodeQueryWhilePrefillInflight) {
  auto fixture = BuildInflightPrefillInferRequest(/*req_id=*/3006);
  auto& req = fixture.infer_req;
  ASSERT_EQ(req->infer_stage, InferStage::kContext);
  ArmDecodePlanning(req);

  PlanningBatch batch;
  batch.dp_batches.resize(1);
  batch.AddRunningReq(/*dp_idx=*/0, req);

  EXPECT_EQ(batch.running_step_token_num, 1u);
  EXPECT_EQ(batch.dp_batches[0].running_step_token_num, 1u);
}

TEST_F(EventDrivenStateTest, PlanningBatchEmptyNotLaunchable) {
  PlanningBatch batch;
  batch.dp_batches.resize(1);
  EXPECT_FALSE(batch.IsLaunchable(/*max_depth=*/2));
}

TEST_F(EventDrivenStateTest, PlanningBatchSkipsStoppedReqInCapacityCheck) {
  auto active_fixture = BuildDecodeReadyInferRequest(/*req_id=*/3003);
  auto stopped_fixture = BuildDecodeReadyInferRequest(/*req_id=*/3004);
  auto& active = active_fixture.infer_req;
  auto& stopped = stopped_fixture.infer_req;
  stopped->Stop();

  PlanningBatch batch;
  batch.dp_batches.resize(1);
  batch.running_req_num = 2;
  batch.dp_batches[0].running_reqs.emplace(active->req_id, active);
  batch.dp_batches[0].running_reqs.emplace(stopped->req_id, stopped);

  ArmDecodePlanning(active);
  active->SetPlanningTask();
  active->LaunchPlanningTask();
  ASSERT_EQ(active->GetInflightCount(), 1u);

  // stopped req 跳过检查；active 在 max_depth=2 下仍可再 launch
  EXPECT_TRUE(batch.IsLaunchable(/*max_depth=*/2));
}

}  // namespace ksana_llm
