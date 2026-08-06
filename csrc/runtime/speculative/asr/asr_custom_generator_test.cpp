/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/speculative/asr/asr_custom_generator.h"

#include <gtest/gtest.h>

#include "runtime/executor_infer_request.h"

namespace ksana_llm {

class AsrCustomGeneratorTest : public testing::Test {
 protected:
  void SetUp() override { asr_custom_generator_ = std::make_unique<AsrCustomGenerator>(); }

 private:
  std::unique_ptr<AsrCustomGenerator> asr_custom_generator_;
};

TEST_F(AsrCustomGeneratorTest, GenerateDraftTest) {
  auto executor_request = std::make_shared<ExecutorInferRequest>();

  // Case 1: Full Match
  executor_request->spec_state.draft_tokens.spec = {2, 3};
  executor_request->spec_state.accepted_tokens = {2, 3};
  asr_custom_generator_->GenerateDraft(executor_request);
  std::vector<int> target_draft_tokens = {};
  std::vector<int> draft_tokens = executor_request->spec_state.draft_tokens.GetDraftTokens();
  EXPECT_EQ(draft_tokens.size(), target_draft_tokens.size());
  EXPECT_EQ(draft_tokens.size(), executor_request->spec_state.draft_tokens.spec.size());

  // Case 2: No Match
  executor_request->spec_state.draft_tokens.spec = {2, 3, 5, 6};
  executor_request->spec_state.accepted_tokens = {};
  executor_request->generated_tokens = {4};
  asr_custom_generator_->GenerateDraft(executor_request);
  target_draft_tokens = {2, 3, 5, 6};
  draft_tokens = executor_request->spec_state.draft_tokens.GetDraftTokens();
  EXPECT_EQ(draft_tokens.size(), target_draft_tokens.size());
  EXPECT_EQ(draft_tokens.size(), executor_request->spec_state.draft_tokens.spec.size());
  for (size_t i = 0; i < target_draft_tokens.size(); ++i) {
    EXPECT_EQ(draft_tokens[i], target_draft_tokens[i]);
  }

  // Case 3: Match in next few positions
  executor_request->spec_state.draft_tokens.spec = {2, 3, 4, 5, 6};
  executor_request->spec_state.accepted_tokens = {};
  executor_request->generated_tokens = {4};
  asr_custom_generator_->GenerateDraft(executor_request);
  target_draft_tokens = {5, 6};
  draft_tokens = executor_request->spec_state.draft_tokens.GetDraftTokens();
  EXPECT_EQ(draft_tokens.size(), target_draft_tokens.size());
  EXPECT_EQ(draft_tokens.size(), executor_request->spec_state.draft_tokens.spec.size());
  for (size_t i = 0; i < target_draft_tokens.size(); ++i) {
    EXPECT_EQ(draft_tokens[i], target_draft_tokens[i]);
  }

  // Case 4: MTP partial match (mismatch), spec should remain unchanged
  executor_request->spec_state.draft_tokens.mtp = {10, 11};
  executor_request->spec_state.draft_tokens.spec = {2, 3, 4};
  executor_request->spec_state.accepted_tokens = {10};  // only 1 MTP accepted (mismatch)
  executor_request->generated_tokens = {99};
  asr_custom_generator_->GenerateDraft(executor_request);
  target_draft_tokens = {10, 11, 2, 3, 4};
  draft_tokens = executor_request->spec_state.draft_tokens.GetDraftTokens();
  EXPECT_EQ(draft_tokens.size(), target_draft_tokens.size());
  EXPECT_EQ(draft_tokens.size(), executor_request->spec_state.draft_tokens.mtp.size() +
                                     executor_request->spec_state.draft_tokens.spec.size());
  for (size_t i = 0; i < target_draft_tokens.size(); ++i) {
    EXPECT_EQ(draft_tokens[i], target_draft_tokens[i]);
  }

  // Case 5: MTP full match, spec partial match with skip
  executor_request->spec_state.draft_tokens.mtp = {10, 11};
  executor_request->spec_state.draft_tokens.spec = {2, 3, 4, 5};
  executor_request->spec_state.accepted_tokens = {10, 11};  // all MTP accepted
  executor_request->generated_tokens = {4};                 // matches spec[2]
  asr_custom_generator_->GenerateDraft(executor_request);
  target_draft_tokens = {10, 11, 5};
  draft_tokens = executor_request->spec_state.draft_tokens.GetDraftTokens();
  EXPECT_EQ(draft_tokens.size(), target_draft_tokens.size());
  EXPECT_EQ(draft_tokens.size(), executor_request->spec_state.draft_tokens.mtp.size() +
                                     executor_request->spec_state.draft_tokens.spec.size());
  for (size_t i = 0; i < target_draft_tokens.size(); ++i) {
    EXPECT_EQ(draft_tokens[i], target_draft_tokens[i]);
  }
}

}  // namespace ksana_llm
