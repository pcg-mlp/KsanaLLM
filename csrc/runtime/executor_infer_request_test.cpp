/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "runtime/executor_infer_request.h"
#include "runtime/sampling_request.h"

namespace ksana_llm {

class ExecutorInferRequestTest : public testing::Test {
 protected:
  void SetUp() override { InitLoguru(); }

  // Build an ExecutorInferRequest primed with the fields GetSamplingRequest reads.
  // model_instance stays null; GetSamplingRequest is nullptr-safe and sets
  // logits_buf to nullptr in that case.
  std::shared_ptr<ExecutorInferRequest> CreateReq(int64_t req_id = 42) {
    auto req = std::make_shared<ExecutorInferRequest>();
    req->req_id = req_id;
    req->logits_custom_length = 7;
    req->input_tokens = {1, 2, 3};
    req->forwarding_tokens = {1, 2, 3, 4};
    req->sampling_token_num = 2;
    req->last_step_token_num = 3;
    req->logits_offset = 5;
    req->sampling_config.num_beams = 1;
    req->sampling_config.topk = 1;
    req->sampling_config.logprobs_num = 0;
    return req;
  }
};

// First call populates static fields; static pointers point into the req.
TEST_F(ExecutorInferRequestTest, StaticFieldsInitializedOnFirstCall) {
  auto req = CreateReq(/*req_id=*/100);

  SamplingRequest* sr = req->GetSamplingRequest(/*enable_main_layers_sampler=*/true);
  ASSERT_NE(sr, nullptr);

  EXPECT_EQ(sr->req_id, 100);
  EXPECT_EQ(sr->logits_custom_length, req->logits_custom_length);
  EXPECT_EQ(sr->sampling_config, &req->sampling_config);
  EXPECT_EQ(sr->forwarding_tokens, &req->forwarding_tokens);
  EXPECT_EQ(sr->sampling_result_tokens, &req->sampling_result_tokens);
  EXPECT_EQ(sr->response, &req->response);
  EXPECT_EQ(sr->ngram_dict, &req->ngram_dict);
  ASSERT_NE(sr->input_tokens, nullptr);
  EXPECT_EQ(sr->input_tokens.get(), &req->input_tokens);
}

// Repeated calls return the same cached instance.
TEST_F(ExecutorInferRequestTest, CachedAcrossCalls) {
  auto req = CreateReq();
  SamplingRequest* first = req->GetSamplingRequest();
  SamplingRequest* second = req->GetSamplingRequest();
  EXPECT_EQ(first, second);
}

// Dynamic fields refresh every call; sampling_result_tokens is cleared.
TEST_F(ExecutorInferRequestTest, DynamicFieldsRefreshEveryCall) {
  auto req = CreateReq();
  SamplingRequest* sr = req->GetSamplingRequest();

  EXPECT_EQ(sr->logits_offset, 5u);
  EXPECT_EQ(sr->sampling_token_num, 2u);
  EXPECT_EQ(sr->last_step_token_num, 3u);

  // Mutate req, call again, expect updated values.
  req->logits_offset = 11;
  req->sampling_token_num = 4;
  req->last_step_token_num = 5;
  req->sampling_result_tokens = {88, 99};

  SamplingRequest* sr2 = req->GetSamplingRequest();
  EXPECT_EQ(sr, sr2);
  EXPECT_EQ(sr2->logits_offset, 11u);
  EXPECT_EQ(sr2->sampling_token_num, 4u);
  EXPECT_EQ(sr2->last_step_token_num, 5u);
  EXPECT_TRUE(sr2->sampling_result_tokens->empty());
}

// enable_main_layers_sampler=true -> apply grammar, disable mtp sampler.
TEST_F(ExecutorInferRequestTest, MainLayersSamplerEnablesGrammarDisablesMtp) {
  auto req = CreateReq();
  SamplingRequest* sr = req->GetSamplingRequest(/*enable_main_layers_sampler=*/true);
  EXPECT_TRUE(sr->apply_grammar_constraint);
  EXPECT_FALSE(sr->enable_mtp_sampler);
}

// enable_main_layers_sampler=false -> mtp path; flags flipped on the SAME cached obj.
TEST_F(ExecutorInferRequestTest, MtpDraftSamplerFlipsFlags) {
  auto req = CreateReq();
  SamplingRequest* first = req->GetSamplingRequest(/*enable_main_layers_sampler=*/true);
  SamplingRequest* second = req->GetSamplingRequest(/*enable_main_layers_sampler=*/false);
  EXPECT_EQ(first, second);
  EXPECT_FALSE(second->apply_grammar_constraint);
  EXPECT_TRUE(second->enable_mtp_sampler);
}

// num_beams>1 lifts logprobs_num and topk to at least num_beams.
TEST_F(ExecutorInferRequestTest, BeamSearchLiftsLogprobsAndTopk) {
  auto req = CreateReq();
  req->sampling_config.num_beams = 4;
  req->sampling_config.logprobs_num = 1;
  req->sampling_config.topk = 2;

  SamplingRequest* sr = req->GetSamplingRequest();
  EXPECT_EQ(sr->sampling_config->logprobs_num, 4);
  EXPECT_EQ(sr->sampling_config->topk, 4);
}

// num_beams>1 does NOT lower already-larger logprobs_num / topk.
TEST_F(ExecutorInferRequestTest, BeamSearchKeepsLargerValues) {
  auto req = CreateReq();
  req->sampling_config.num_beams = 2;
  req->sampling_config.logprobs_num = 10;
  req->sampling_config.topk = 20;

  SamplingRequest* sr = req->GetSamplingRequest();
  EXPECT_EQ(sr->sampling_config->logprobs_num, 10);
  EXPECT_EQ(sr->sampling_config->topk, 20);
}

// Null model_instance -> logits_buf falls back to nullptr, no crash.
TEST_F(ExecutorInferRequestTest, NullModelInstanceYieldsNullLogitsBuf) {
  auto req = CreateReq();
  SamplingRequest* sr = req->GetSamplingRequest();
  EXPECT_EQ(sr->logits_buf, nullptr);
}

// input_tokens / logprobs are aliasing shared_ptrs with empty owner.
// They point into the req but hold no ownership (use_count() == 0).
TEST_F(ExecutorInferRequestTest, AliasingSharedPtrsAreNonOwning) {
  auto req = CreateReq();
  SamplingRequest* sr = req->GetSamplingRequest();

  // input_tokens: aliasing shared_ptr with empty owner
  EXPECT_EQ(sr->input_tokens.use_count(), 0);
  EXPECT_EQ(sr->input_tokens.get(), &req->input_tokens);
  EXPECT_EQ(sr->input_tokens->size(), req->input_tokens.size());

  // logprobs: same non-owning aliasing pattern
  EXPECT_EQ(sr->logprobs.use_count(), 0);
  EXPECT_EQ(sr->logprobs.get(), &req->logprobs);
}

}  // namespace ksana_llm
