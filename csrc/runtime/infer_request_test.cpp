/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "runtime/infer_request.h"
#include "runtime/infer_stage.h"
#include "utils/request.h"

namespace ksana_llm {

class InferRequestTest : public testing::Test {
 protected:
  void SetUp() override { InitLoguru(); }

  // Helper to create an InferRequest via Request (which requires KsanaPythonInput).
  std::shared_ptr<InferRequest> CreateInferRequest(const std::vector<int>& input_tokens,
                                                   const std::vector<int>& stop_token_ids = {}) {
    auto python_input = std::make_shared<KsanaPythonInput>();
    python_input->input_tokens = input_tokens;
    python_input->sampling_config.stop_token_ids = stop_token_ids;

    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    request_ = std::make_shared<Request>(python_input, req_ctx);
    python_input_ = python_input;  // prevent KsanaPythonInput from being destroyed

    auto infer_req = std::make_shared<InferRequest>(request_, 0);
    return infer_req;
  }

  // Set up an inflight task for the infer request.
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

  // Set up an inflight decode task.
  void SetupInflightDecodeTask(std::shared_ptr<InferRequest>& req, size_t generated_token_num = 1,
                               size_t draft_token_num = 0) {
    ScheduleTaskWorkload workload;
    workload.generated_token_num = generated_token_num;
    workload.draft_token_num = draft_token_num;
    workload.sampling_token_num = generated_token_num + draft_token_num;
    req->SetPlanningWorkload(workload);
    req->SetPlanningTask();
    req->LaunchPlanningTask();
  }

  std::shared_ptr<KsanaPythonInput> python_input_;
  std::shared_ptr<Request> request_;
};

// Test: Intermediate prefill step keeps infer_stage as kContext and clears generated_tokens/draft_tokens.
TEST_F(InferRequestTest, IntermediatePrefillKeepsContextStage) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5, 6});
  // Initialize prefilling state: output_tokens = input_tokens initially
  req->output_tokens = {1, 2, 3, 4, 5, 6};
  req->ResetPrefillingTokens();

  // Simulate first chunk of prefill: 3 out of 6 tokens (intermediate step).
  SetupInflightPrefillTask(req, /*prefill_token_num=*/3, /*prefill_start_offset=*/0);

  // Simulate some stale data that might be left from previous operations.
  req->generated_tokens = {99};
  req->draft_tokens.mtp = {88, 77};

  size_t output_tokens_size_before = req->output_tokens.size();
  req->UpdateAfterInflightTaskFinished();

  // After intermediate prefill: should stay in kContext stage
  EXPECT_EQ(req->infer_stage, InferStage::kContext);
  // generated_tokens and draft_tokens should be cleared
  EXPECT_TRUE(req->generated_tokens.empty());
  EXPECT_EQ(req->draft_tokens.size(), 0);
  // output_tokens must be unchanged
  EXPECT_EQ(req->output_tokens.size(), output_tokens_size_before);
}

// Test: Last prefill step transitions infer_stage to kDecode and appends to output_tokens.
TEST_F(InferRequestTest, LastPrefillStepTransitionsToDecode) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5, 6});
  req->output_tokens = {1, 2, 3, 4, 5, 6};
  req->ResetPrefillingTokens();

  // First intermediate prefill step: 3 out of 6
  SetupInflightPrefillTask(req, /*prefill_token_num=*/3, /*prefill_start_offset=*/0);
  req->UpdateAfterInflightTaskFinished();
  req->ResetInflightTask();

  // Second (last) prefill step: remaining 3 tokens
  SetupInflightPrefillTask(req, /*prefill_token_num=*/3, /*prefill_start_offset=*/3);

  // Simulate generation result for the last prefill step
  req->generated_tokens = {100};

  req->UpdateAfterInflightTaskFinished();

  // After last prefill: should transition to kDecode
  EXPECT_EQ(req->infer_stage, InferStage::kDecode);
  // output_tokens should contain input + generated token
  EXPECT_EQ(req->output_tokens.back(), 100);
}

// Test: Decode step with draft tokens accepted updates forwarding_tokens and output_tokens correctly.
TEST_F(InferRequestTest, DecodeStepWithDraftTokensAccepted) {
  auto req = CreateInferRequest({1, 2, 3}, /*stop_token_ids=*/{});
  req->output_tokens = {1, 2, 3};
  req->ResetPrefillingTokens();

  // Complete the prefill phase first
  SetupInflightPrefillTask(req, /*prefill_token_num=*/3, /*prefill_start_offset=*/0);
  req->generated_tokens = {10};
  req->UpdateAfterInflightTaskFinished();
  req->ResetInflightTask();

  EXPECT_EQ(req->infer_stage, InferStage::kDecode);

  // Now set up a decode step with draft tokens
  // forwarding_tokens = input(3) + generated(1) + draft(2) = [1,2,3,10, 11,12]
  req->forwarding_tokens = {1, 2, 3, 10, 11, 12};
  req->forwarding_tokens_draft_num = 2;
  req->draft_tokens.mtp = {11, 12};

  SetupInflightDecodeTask(req, /*generated_token_num=*/1, /*draft_token_num=*/2);

  // Simulate: draft tokens 11,12 were forwarded, model verified them:
  // accepted_tokens = {11} (draft token 11 accepted), generated_tokens = {20} (new token)
  req->accepted_tokens = {11};
  req->generated_tokens = {20};

  size_t forwarding_tokens_size_before = req->forwarding_tokens.size();

  req->UpdateAfterInflightTaskFinished();

  // forwarding_tokens should be resized: original(6) - draft_num(2) + accepted(1) = 5
  EXPECT_EQ(req->forwarding_tokens.size(), forwarding_tokens_size_before - 2 + 1);
  // output_tokens should contain accepted + generated tokens appended
  // output_tokens was [1,2,3,10], now should have 11 (accepted) and 20 (generated) appended
  EXPECT_GE(req->output_tokens.size(), 2);
  size_t n = req->output_tokens.size();
  EXPECT_EQ(req->output_tokens[n - 2], 11);
  EXPECT_EQ(req->output_tokens[n - 1], 20);
  // forwarding_tokens should end with accepted tokens content
  EXPECT_EQ(req->forwarding_tokens.back(), 11);  // accepted token
}

// Test: EOS token detection in decode step
TEST_F(InferRequestTest, EosTokenDetectedInDecodeStep) {
  int eos_token = 999;
  auto req = CreateInferRequest({1, 2, 3}, /*stop_token_ids=*/{eos_token});
  req->output_tokens = {1, 2, 3};
  req->ResetPrefillingTokens();

  // Complete prefill
  SetupInflightPrefillTask(req, /*prefill_token_num=*/3, /*prefill_start_offset=*/0);
  req->generated_tokens = {10};
  req->UpdateAfterInflightTaskFinished();
  req->ResetInflightTask();

  // Decode step generating EOS
  SetupInflightDecodeTask(req, /*generated_token_num=*/1, /*draft_token_num=*/0);
  req->generated_tokens = {eos_token};
  req->UpdateAfterInflightTaskFinished();

  EXPECT_TRUE(req->IsEosGenerated());
  EXPECT_EQ(req->output_tokens.back(), eos_token);
}

// Test: IsSplitPrefillStep returns true for first chunk (offset=0, num < total).
TEST_F(InferRequestTest, IsSplitPrefillStep_FirstChunk) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5, 6});
  req->output_tokens = {1, 2, 3, 4, 5, 6};
  req->ResetPrefillingTokens();

  // Prefill 2 out of 6 tokens from offset 0 → intermediate step.
  SetupInflightPrefillTask(req, /*prefill_token_num=*/2, /*prefill_start_offset=*/0);
  EXPECT_TRUE(req->IsSplitPrefillStep());
}

// Test: IsSplitPrefillStep returns true for middle chunk (offset > 0, offset + num < total).
TEST_F(InferRequestTest, IsSplitPrefillStep_MiddleChunk) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5, 6});
  req->output_tokens = {1, 2, 3, 4, 5, 6};
  req->ResetPrefillingTokens();

  // Simulate: first chunk already done, now prefill 2 tokens from offset 2 (total=6).
  // 2 + 2 = 4 < 6 → still intermediate.
  SetupInflightPrefillTask(req, /*prefill_token_num=*/2, /*prefill_start_offset=*/0);
  req->UpdateAfterInflightTaskFinished();
  req->ResetInflightTask();

  SetupInflightPrefillTask(req, /*prefill_token_num=*/2, /*prefill_start_offset=*/2);
  EXPECT_TRUE(req->IsSplitPrefillStep());
}

// Test: IsSplitPrefillStep returns false when offset + num == total (last step, non-zero offset).
TEST_F(InferRequestTest, IsSplitPrefillStep_LastChunkWithOffset) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5, 6});
  req->output_tokens = {1, 2, 3, 4, 5, 6};
  req->ResetPrefillingTokens();

  // First chunk: 4 out of 6.
  SetupInflightPrefillTask(req, /*prefill_token_num=*/4, /*prefill_start_offset=*/0);
  req->UpdateAfterInflightTaskFinished();
  req->ResetInflightTask();

  // Last chunk: offset=4, num=2 → 4+2 == 6 → NOT a split step.
  SetupInflightPrefillTask(req, /*prefill_token_num=*/2, /*prefill_start_offset=*/4);
  EXPECT_FALSE(req->IsSplitPrefillStep());
}

// Test: IsSplitPrefillStep returns false when all tokens are prefilled in a single step.
TEST_F(InferRequestTest, IsSplitPrefillStep_SingleStepPrefill) {
  auto req = CreateInferRequest({1, 2, 3});
  req->output_tokens = {1, 2, 3};
  req->ResetPrefillingTokens();

  // Prefill all 3 tokens at once → not a split step.
  SetupInflightPrefillTask(req, /*prefill_token_num=*/3, /*prefill_start_offset=*/0);
  EXPECT_FALSE(req->IsSplitPrefillStep());
}

}  // namespace ksana_llm
