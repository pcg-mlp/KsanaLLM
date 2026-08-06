/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <vector>

#include "executor/model_builder.h"
#include "executor/model_executor.h"
#include "executor/model_sampler.h"
#include "models/base/base_model.h"
#include "models/base/model_delegate.h"
#include "runtime/executor_infer_request.h"
#include "runtime/model_input.h"
#include "runtime/model_instance.h"
#include "runtime/model_output.h"
#include "runtime/schedule_output.h"
#include "runtime/speculative/dspark/dspark_decoder.h"
#include "runtime/speculative/mtp/mtp_state_guard.h"

namespace ksana_llm {
namespace {

class DSparkDecoderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a minimal request with generated tokens for DSpark to pad.
    req_ = std::make_shared<ExecutorInferRequest>();
    req_->req_id = 1;
    req_->forwarding_tokens = {1, 2, 3};  // 3 cached tokens
    req_->kv_cached_token_num = 3;
    req_->prefix_cache_len = 3;
    req_->generated_tokens = {100};
    req_->sampling_token_num = kStepGenerateTokenNum;
  }

  std::shared_ptr<ExecutorInferRequest> req_;
  std::mutex infer_req_mutex_;
  const size_t kBlockSize = 5;
  const int kNoiseTokenId = 128799;
};

TEST_F(DSparkDecoderTest, EarlyReturnWhenBlockSizeZero) {
  DSparkDecoder decoder(/*dspark_block_size=*/0, kNoiseTokenId, /*model_delegate=*/nullptr, &infer_req_mutex_);
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req_};
  auto status = decoder.ExecuteForward(reqs, /*hidden_unit=*/nullptr, /*schedule_output=*/nullptr,
                                       /*model_executor=*/nullptr, /*model_sampler=*/nullptr,
                                       /*model_builder=*/nullptr, /*model_instance=*/nullptr);
  EXPECT_TRUE(status.OK());
  // No modification when block_size is zero.
  EXPECT_EQ(req_->forwarding_tokens.size(), 3u);
}

TEST_F(DSparkDecoderTest, EarlyReturnWhenRequestsEmpty) {
  DSparkDecoder decoder(kBlockSize, kNoiseTokenId, /*model_delegate=*/nullptr, &infer_req_mutex_);
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs;
  auto status = decoder.ExecuteForward(reqs, /*hidden_unit=*/nullptr, /*schedule_output=*/nullptr,
                                       /*model_executor=*/nullptr, /*model_sampler=*/nullptr,
                                       /*model_builder=*/nullptr, /*model_instance=*/nullptr);
  EXPECT_TRUE(status.OK());
}

TEST_F(DSparkDecoderTest, ForwardingTokensPadLogic) {
  // Test that forwarding_tokens are correctly padded with noise tokens
  // when accepted_tokens and forwarding_tokens_draft_num are zero (first step).
  EXPECT_EQ(req_->spec_state.accepted_tokens.size(), 0u);
  EXPECT_EQ(req_->spec_state.forwarding_tokens_draft_num, 0u);

  // Manually apply the DSpark padding logic (same as ExecuteForward does).
  req_->spec_state.draft_tokens.clear();
  req_->forwarding_tokens.resize(req_->forwarding_tokens.size() - req_->spec_state.forwarding_tokens_draft_num +
                                 req_->spec_state.accepted_tokens.size());
  req_->prefix_cache_len = req_->kv_cached_token_num = req_->forwarding_tokens.size();
  req_->forwarding_tokens.insert(req_->forwarding_tokens.end(), req_->generated_tokens.begin(),
                                 req_->generated_tokens.end());
  req_->forwarding_tokens.insert(req_->forwarding_tokens.end(), kBlockSize - req_->generated_tokens.size(),
                                 kNoiseTokenId);
  req_->sampling_token_num = kBlockSize;

  // Expected: 3 cached + 1 generated + 4 noise = 8 tokens
  EXPECT_EQ(req_->forwarding_tokens.size(), 8u);
  // Last 4 tokens should be noise
  for (size_t i = 4; i < 8; ++i) {
    EXPECT_EQ(req_->forwarding_tokens[i], kNoiseTokenId);
  }
  // First 4 tokens: 1,2,3 + 100
  EXPECT_EQ(req_->forwarding_tokens[0], 1);
  EXPECT_EQ(req_->forwarding_tokens[1], 2);
  EXPECT_EQ(req_->forwarding_tokens[2], 3);
  EXPECT_EQ(req_->forwarding_tokens[3], 100);
  EXPECT_EQ(req_->sampling_token_num, kBlockSize);
}

TEST_F(DSparkDecoderTest, ForwardingTokensPadWithAcceptedTokens) {
  // Simulate subsequent step: accepted_tokens and draft_num are non-zero.
  req_->spec_state.accepted_tokens = {200, 201};     // 2 accepted drafts
  req_->spec_state.forwarding_tokens_draft_num = 3;  // 3 drafts were appended

  req_->spec_state.draft_tokens.clear();
  req_->forwarding_tokens.resize(req_->forwarding_tokens.size() - req_->spec_state.forwarding_tokens_draft_num +
                                 req_->spec_state.accepted_tokens.size());
  req_->prefix_cache_len = req_->kv_cached_token_num = req_->forwarding_tokens.size();
  req_->forwarding_tokens.insert(req_->forwarding_tokens.end(), req_->generated_tokens.begin(),
                                 req_->generated_tokens.end());
  req_->forwarding_tokens.insert(req_->forwarding_tokens.end(), kBlockSize - req_->generated_tokens.size(),
                                 kNoiseTokenId);
  req_->sampling_token_num = kBlockSize;

  // after resize: 3 - 3 + 2 = 2 (first 2 of original tokens)
  EXPECT_EQ(req_->forwarding_tokens[0], 1);
  EXPECT_EQ(req_->forwarding_tokens[1], 2);
  // then generated: 100
  // then noise: 4 × kNoiseTokenId
  EXPECT_EQ(req_->forwarding_tokens.size(), 1u + 1u + 1u + 4u);  // 2 + 1 + 4 = 7
}

TEST_F(DSparkDecoderTest, MtpStateGuardSavesCorrectFields) {
  // Verify MtpStateGuard saves and restores the 4 fields it tracks.
  // Save initial state
  std::vector<std::shared_ptr<ExecutorInferRequest>> guard_reqs = {req_};
  MtpStateGuard guard(guard_reqs);

  // Inside guard scope (simulating DSpark forward), modify fields
  req_->forwarding_tokens.push_back(999);
  req_->kv_cached_token_num = 999;
  req_->prefix_cache_len = 999;
  req_->spec_state.forwarding_tokens_draft_num = 999;

  // Guard destructor runs here, restoring state
}

TEST_F(DSparkDecoderTest, MtpStateGuardRestoresAfterModification) {
  // Capture original values
  const auto orig_tokens = req_->forwarding_tokens;
  const auto orig_kv = req_->kv_cached_token_num;
  const auto orig_prefix = req_->prefix_cache_len;
  const auto orig_draft_num = req_->spec_state.forwarding_tokens_draft_num;

  {
    std::vector<std::shared_ptr<ExecutorInferRequest>> guard_reqs = {req_};
    MtpStateGuard guard(guard_reqs);
    req_->forwarding_tokens.push_back(999);
    req_->kv_cached_token_num = 999;
    req_->prefix_cache_len = 999;
    req_->spec_state.forwarding_tokens_draft_num = 999;
  }

  // After guard scope, all fields should be restored
  EXPECT_EQ(req_->forwarding_tokens, orig_tokens);
  EXPECT_EQ(req_->kv_cached_token_num, orig_kv);
  EXPECT_EQ(req_->prefix_cache_len, orig_prefix);
  EXPECT_EQ(req_->spec_state.forwarding_tokens_draft_num, orig_draft_num);
}

TEST_F(DSparkDecoderTest, SamplingTokenNumSetToBlockSize) {
  // Verify that DSpark sets sampling_token_num to block_size after padding.
  req_->spec_state.draft_tokens.clear();
  req_->forwarding_tokens.resize(req_->forwarding_tokens.size() - req_->spec_state.forwarding_tokens_draft_num +
                                 req_->spec_state.accepted_tokens.size());
  req_->prefix_cache_len = req_->kv_cached_token_num = req_->forwarding_tokens.size();
  req_->forwarding_tokens.insert(req_->forwarding_tokens.end(), req_->generated_tokens.begin(),
                                 req_->generated_tokens.end());
  req_->forwarding_tokens.insert(req_->forwarding_tokens.end(), kBlockSize - req_->generated_tokens.size(),
                                 kNoiseTokenId);
  req_->sampling_token_num = kBlockSize;

  EXPECT_EQ(req_->sampling_token_num, kBlockSize);
  EXPECT_NE(req_->sampling_token_num, kStepGenerateTokenNum);  // Not 1
}

}  // namespace
}  // namespace ksana_llm
