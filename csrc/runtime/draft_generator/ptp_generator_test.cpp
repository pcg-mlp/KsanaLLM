/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/draft_generator/ptp_generator.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "runtime/executor_infer_request.h"

namespace ksana_llm {

// Test for PtpGenerator speculative decoding logic
class PtpGeneratorTest : public testing::Test {
 protected:
  void SetUp() override {
    // PtpGenerator with ptp_step_num=3, ptp_token_id=999
    generator_ = std::make_shared<PtpGenerator>(3, 999);
  }

  std::shared_ptr<ExecutorInferRequest> CreateMockInferRequest(const std::vector<int>& draft_tokens_ptp,
                                                               const std::vector<int>& sampling_result_tokens) {
    auto executor_req = std::make_shared<ExecutorInferRequest>();

    executor_req->req_id = 1;
    executor_req->input_tokens = {1, 2, 3};
    executor_req->output_tokens = {};
    executor_req->finished = false;
    executor_req->aborted = false;

    executor_req->draft_tokens.ptp = draft_tokens_ptp;
    executor_req->sampling_result_tokens = sampling_result_tokens;

    return executor_req;
  }

  std::shared_ptr<PtpGenerator> generator_;
};

// Test: GenerateDraft generates placeholder tokens
TEST_F(PtpGeneratorTest, GenerateDraft) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 1;

  generator_->GenerateDraft(req);

  EXPECT_EQ(req->draft_tokens.ptp.size(), 3);
  EXPECT_EQ(req->draft_tokens.ptp[0], 999);
  EXPECT_EQ(req->draft_tokens.ptp[1], 999);
  EXPECT_EQ(req->draft_tokens.ptp[2], 999);
}

// Test: GenerateDraft appends placeholders after existing saved guess tokens
TEST_F(PtpGeneratorTest, GenerateDraft_AppendPlaceholder) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 1;
  // Simulate saved guess tokens from previous FilterDraftTokens
  req->draft_tokens.ptp = {10, 20};

  generator_->GenerateDraft(req);

  // Should append 3 placeholders after the 2 saved tokens
  ASSERT_EQ(req->draft_tokens.ptp.size(), 5);
  EXPECT_EQ(req->draft_tokens.ptp[0], 10);
  EXPECT_EQ(req->draft_tokens.ptp[1], 20);
  EXPECT_EQ(req->draft_tokens.ptp[2], 999);
  EXPECT_EQ(req->draft_tokens.ptp[3], 999);
  EXPECT_EQ(req->draft_tokens.ptp[4], 999);
}

// Test: Pure guess round (all placeholders, verify_count=0)
// draft_tokens.ptp = [999, 999, 999]
// sampling_result_tokens = [s0, s1, s2, s3]
// Result: generated=[s0], saved=[s1,s2,s3], accepted=[]
TEST_F(PtpGeneratorTest, GuessRound) {
  auto req = CreateMockInferRequest({999, 999, 999}, {10, 20, 30, 40});

  size_t accepted = generator_->FilterDraftTokens(req);

  EXPECT_EQ(accepted, 0);
  EXPECT_TRUE(req->accepted_tokens.empty());
  // generated_tokens = [s0]
  ASSERT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 10);
  // Save [s1, s2, s3] for next step verification
  ASSERT_EQ(req->draft_tokens.ptp.size(), 3);
  EXPECT_EQ(req->draft_tokens.ptp[0], 20);
  EXPECT_EQ(req->draft_tokens.ptp[1], 30);
  EXPECT_EQ(req->draft_tokens.ptp[2], 40);
  EXPECT_TRUE(req->sampling_result_tokens.empty());
}

// Test: Pure guess round with logprobs - only keep generated token's logprob
TEST_F(PtpGeneratorTest, GuessRound_Logprobs) {
  auto req = CreateMockInferRequest({999, 999, 999}, {10, 20, 30, 40});
  req->logprobs = {{{10, 0.5f}}, {{20, 0.3f}}, {{30, 0.2f}}, {{40, 0.1f}}};

  generator_->FilterDraftTokens(req);

  // Only keep logprob for generated token (s0), pop 3 placeholder logprobs
  ASSERT_EQ(req->logprobs.size(), 1);
  EXPECT_EQ(req->logprobs[0][0].first, 10);
}

// Test: Verification all accepted
// draft_tokens.ptp = [10, 20, 999, 999, 999] (verify: 10,20; placeholder: 999,999,999)
// sampling_result_tokens = [10, 20, 30, 40, 50, 60]
// s0=10==10, s1=20==20: all verify pass
// Result: accepted=[10,20], generated=[30], saved=[40,50,60]
TEST_F(PtpGeneratorTest, VerifyAllAccepted) {
  auto req = CreateMockInferRequest({10, 20, 999, 999, 999}, {10, 20, 30, 40, 50, 60});

  size_t accepted = generator_->FilterDraftTokens(req);

  EXPECT_EQ(accepted, 2);
  ASSERT_EQ(req->accepted_tokens.size(), 2);
  EXPECT_EQ(req->accepted_tokens[0], 10);
  EXPECT_EQ(req->accepted_tokens[1], 20);
  // generated_tokens = [30] (first placeholder position)
  ASSERT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 30);
  // Save [40, 50, 60] for next step verification
  ASSERT_EQ(req->draft_tokens.ptp.size(), 3);
  EXPECT_EQ(req->draft_tokens.ptp[0], 40);
  EXPECT_EQ(req->draft_tokens.ptp[1], 50);
  EXPECT_EQ(req->draft_tokens.ptp[2], 60);
  EXPECT_TRUE(req->sampling_result_tokens.empty());
}

// Test: Verification all accepted with logprobs
TEST_F(PtpGeneratorTest, VerifyAllAccepted_Logprobs) {
  auto req = CreateMockInferRequest({10, 20, 999, 999, 999}, {10, 20, 30, 40, 50, 60});
  // logprobs[i] corresponds to sampling_result_tokens[i]
  req->logprobs = {{{10, 0.9f}}, {{20, 0.8f}}, {{30, 0.7f}}, {{40, 0.6f}}, {{50, 0.5f}}, {{60, 0.4f}}};

  generator_->FilterDraftTokens(req);

  // Keep: 2 (accepted) + 1 (generated) = 3 logprobs, pop 3 (placeholder)
  ASSERT_EQ(req->logprobs.size(), 3);
  EXPECT_EQ(req->logprobs[0][0].first, 10);
  EXPECT_EQ(req->logprobs[1][0].first, 20);
  EXPECT_EQ(req->logprobs[2][0].first, 30);
}

// Test: Verification partial accepted (first matches, second doesn't)
// draft_tokens.ptp = [10, 20, 999, 999, 999]
// sampling_result_tokens = [10, 99, 30, 40, 50, 60]
// s0=10==10, s1=99!=20: partial
// Result: accepted=[10], generated=[99], draft_tokens cleared
TEST_F(PtpGeneratorTest, VerifyPartialAccepted) {
  auto req = CreateMockInferRequest({10, 20, 999, 999, 999}, {10, 99, 30, 40, 50, 60});

  size_t accepted = generator_->FilterDraftTokens(req);

  EXPECT_EQ(accepted, 1);
  ASSERT_EQ(req->accepted_tokens.size(), 1);
  EXPECT_EQ(req->accepted_tokens[0], 10);
  // generated_tokens = [99] (the mismatched sampling result)
  ASSERT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 99);
  // All draft_tokens cleared (truncate everything including placeholders)
  EXPECT_EQ(req->draft_tokens.size(), 0);
  EXPECT_TRUE(req->sampling_result_tokens.empty());
}

// Test: Verification partial accepted with logprobs
TEST_F(PtpGeneratorTest, VerifyPartialAccepted_Logprobs) {
  auto req = CreateMockInferRequest({10, 20, 999, 999, 999}, {10, 99, 30, 40, 50, 60});
  req->logprobs = {{{10, 0.9f}}, {{99, 0.8f}}, {{30, 0.7f}}, {{40, 0.6f}}, {{50, 0.5f}}, {{60, 0.4f}}};

  generator_->FilterDraftTokens(req);

  // Keep: 1 (accepted) + 1 (generated) = 2, pop 4 (unaccepted)
  ASSERT_EQ(req->logprobs.size(), 2);
  EXPECT_EQ(req->logprobs[0][0].first, 10);
  EXPECT_EQ(req->logprobs[1][0].first, 99);
}

// Test: Verification none accepted (first token doesn't match)
// draft_tokens.ptp = [10, 20, 999, 999, 999]
// sampling_result_tokens = [99, 88, 30, 40, 50, 60]
// s0=99!=10: none accepted
// Result: accepted=[], generated=[99], draft_tokens cleared
TEST_F(PtpGeneratorTest, VerifyNoneAccepted) {
  auto req = CreateMockInferRequest({10, 20, 999, 999, 999}, {99, 88, 30, 40, 50, 60});

  size_t accepted = generator_->FilterDraftTokens(req);

  EXPECT_EQ(accepted, 0);
  EXPECT_TRUE(req->accepted_tokens.empty());
  ASSERT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 99);
  EXPECT_EQ(req->draft_tokens.size(), 0);
  EXPECT_TRUE(req->sampling_result_tokens.empty());
}

// Test: Verification none accepted with logprobs
TEST_F(PtpGeneratorTest, VerifyNoneAccepted_Logprobs) {
  auto req = CreateMockInferRequest({10, 20, 999, 999, 999}, {99, 88, 30, 40, 50, 60});
  req->logprobs = {{{99, 0.9f}}, {{88, 0.8f}}, {{30, 0.7f}}, {{40, 0.6f}}, {{50, 0.5f}}, {{60, 0.4f}}};

  generator_->FilterDraftTokens(req);

  // Keep: 0 (accepted) + 1 (generated) = 1, pop 5
  ASSERT_EQ(req->logprobs.size(), 1);
  EXPECT_EQ(req->logprobs[0][0].first, 99);
}

// Test: Mixed mode (MTP + PTP) falls back to DefaultFilterDraftTokens
TEST_F(PtpGeneratorTest, MixedMode_Fallback) {
  // draft_tokens: mtp={100, 200} + ptp={999, 999, 999} → total 5
  // sampling_result_tokens needs 6 elements (5 + kStepGenerateTokenNum)
  auto req = CreateMockInferRequest({999, 999, 999}, {10, 20, 30, 40, 50, 60});
  req->draft_tokens.mtp = {100, 200};  // Has MTP tokens

  size_t accepted = generator_->FilterDraftTokens(req);

  // GetDraftTokens() = {100, 200, 999, 999, 999}, sampling[0]=10 != 100 → draft_hit_num=0
  EXPECT_EQ(accepted, 0);
  EXPECT_TRUE(req->accepted_tokens.empty());
  ASSERT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 10);
  EXPECT_TRUE(req->sampling_result_tokens.empty());
}

// Test: Mixed mode with Trie tokens also falls back to DefaultFilterDraftTokens
TEST_F(PtpGeneratorTest, MixedMode_TrieFallback) {
  // draft_tokens: spec={300} + ptp={999, 999} → total 3
  // sampling_result_tokens needs 4 elements (3 + kStepGenerateTokenNum)
  auto req = CreateMockInferRequest({999, 999}, {10, 20, 30, 40});
  req->draft_tokens.spec = {300};  // Has Trie tokens

  size_t accepted = generator_->FilterDraftTokens(req);

  // GetDraftTokens() = {300, 999, 999}, sampling[0]=10 != 300 → draft_hit_num=0
  EXPECT_EQ(accepted, 0);
  EXPECT_TRUE(req->accepted_tokens.empty());
  ASSERT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 10);
  EXPECT_TRUE(req->sampling_result_tokens.empty());
}

// Test: Multi-step flow simulation (guess -> verify_all_pass -> verify_partial)
TEST_F(PtpGeneratorTest, MultiStepFlow) {
  // Step 1: Pure guess round
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 1;
  req->input_tokens = {1, 2, 3};
  req->finished = false;
  req->aborted = false;

  // GenerateDraft: append 3 placeholders
  generator_->GenerateDraft(req);
  ASSERT_EQ(req->draft_tokens.ptp.size(), 3);

  // Simulate forward + sample: sampling_result_tokens = [s0, s1, s2, s3]
  req->sampling_result_tokens = {100, 200, 300, 400};

  size_t accepted = generator_->FilterDraftTokens(req);
  EXPECT_EQ(accepted, 0);
  ASSERT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 100);
  // Saved guesses: [200, 300, 400]
  ASSERT_EQ(req->draft_tokens.ptp.size(), 3);
  EXPECT_EQ(req->draft_tokens.ptp[0], 200);
  EXPECT_EQ(req->draft_tokens.ptp[1], 300);
  EXPECT_EQ(req->draft_tokens.ptp[2], 400);

  // Step 2: GenerateDraft appends 3 more placeholders
  generator_->GenerateDraft(req);
  // draft_tokens.ptp = [200, 300, 400, 999, 999, 999]
  ASSERT_EQ(req->draft_tokens.ptp.size(), 6);

  // Simulate forward + sample with all verify passing
  // sampling_result_tokens = [200, 300, 400, 500, 600, 700, 800]
  req->sampling_result_tokens = {200, 300, 400, 500, 600, 700, 800};

  accepted = generator_->FilterDraftTokens(req);
  EXPECT_EQ(accepted, 3);  // All 3 verify tokens matched
  ASSERT_EQ(req->accepted_tokens.size(), 3);
  EXPECT_EQ(req->accepted_tokens[0], 200);
  EXPECT_EQ(req->accepted_tokens[1], 300);
  EXPECT_EQ(req->accepted_tokens[2], 400);
  ASSERT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 500);  // First placeholder result
  // Saved: [600, 700, 800]
  ASSERT_EQ(req->draft_tokens.ptp.size(), 3);

  // Step 3: GenerateDraft again, then verify with partial failure
  generator_->GenerateDraft(req);
  // draft_tokens.ptp = [600, 700, 800, 999, 999, 999]
  ASSERT_EQ(req->draft_tokens.ptp.size(), 6);

  // First two match, third doesn't
  req->sampling_result_tokens = {600, 700, 999, 50, 60, 70, 80};

  accepted = generator_->FilterDraftTokens(req);
  EXPECT_EQ(accepted, 2);  // Only first 2 matched
  ASSERT_EQ(req->accepted_tokens.size(), 2);
  ASSERT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 999);  // Mismatch position sampling result
  // Draft tokens cleared on failure
  EXPECT_EQ(req->draft_tokens.size(), 0);
}

}  // namespace ksana_llm
