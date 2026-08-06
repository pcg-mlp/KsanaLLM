/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/speculative/ptp/ptp_decoder_adapter.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "runtime/speculative/ptp/ptp_generator.h"
#include "runtime/executor_infer_request.h"

namespace ksana_llm {

class PtpDecoderAdapterTest : public testing::Test {
 protected:
  void SetUp() override {
    generator_ = std::make_shared<PtpGenerator>(/*ptp_step_num=*/3, /*ptp_token_id=*/999);
    adapter_ = std::make_unique<PtpDecoderAdapter>(generator_);
  }

  std::shared_ptr<PtpGenerator> generator_;
  std::unique_ptr<PtpDecoderAdapter> adapter_;
};

TEST_F(PtpDecoderAdapterTest, GetNameReturnsPTP) { EXPECT_STREQ(adapter_->GetName(), "PTP"); }

TEST_F(PtpDecoderAdapterTest, GetSchedulingInfoDelegatesToGenerator) {
  // PtpGenerator with ptp_step_num=3: extra_token_capacity = workload_multiplier = 2*3 = 6.
  auto info = adapter_->GetSchedulingInfo();
  EXPECT_EQ(info.extra_token_capacity, 6u);
  EXPECT_EQ(info.workload_multiplier, 6u);
}

// BeforeForward delegates to generator_->GenerateDraft() for every request in ctx.reqs.
TEST_F(PtpDecoderAdapterTest, BeforeForwardGeneratesDraftForAllRequests) {
  auto req1 = std::make_shared<ExecutorInferRequest>();
  req1->req_id = 1;
  req1->sampling_config.ptp_token_num = -1;  // use generator's default ptp_step_num
  auto req2 = std::make_shared<ExecutorInferRequest>();
  req2->req_id = 2;
  req2->sampling_config.ptp_token_num = -1;
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req1, req2};

  SpecContext ctx = SpecContext::ForDraftGen(&reqs, nullptr);
  adapter_->BeforeForward(ctx);

  EXPECT_EQ(req1->spec_state.draft_tokens.ptp.size(), 3u);
  EXPECT_EQ(req2->spec_state.draft_tokens.ptp.size(), 3u);
}

// BeforeForward is a no-op (does not crash) when ctx.reqs is null.
TEST_F(PtpDecoderAdapterTest, BeforeForwardNoopWhenReqsNull) {
  SpecContext ctx;
  ctx.reqs = nullptr;
  adapter_->BeforeForward(ctx);
  SUCCEED();
}

// ExecuteForward and AfterForward are declared no-ops in the header (inline {}).
// Verify they don't crash and produce no observable side effects.
TEST_F(PtpDecoderAdapterTest, ExecuteForwardAndAfterForwardAreNoOps) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 1;
  req->spec_state.draft_tokens.ptp = {999, 999};
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};
  SpecContext ctx = SpecContext::ForDraftGen(&reqs, nullptr);

  adapter_->ExecuteForward(ctx);
  adapter_->AfterForward(ctx);

  // draft_tokens are untouched by these two no-op methods.
  EXPECT_EQ(req->spec_state.draft_tokens.ptp.size(), 2u);
}

// Verify delegates to generator_->FilterDraftTokens() for every request.
TEST_F(PtpDecoderAdapterTest, VerifyDelegatesToGeneratorFilterDraftTokens) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 1;
  // Cold-start case per PtpGenerator::FilterDraftTokens: sampling_result_tokens.size() <= 1
  // triggers the pass-through path (no draft/mtp/spec state needed).
  req->sampling_result_tokens = {42};
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};
  SpecContext ctx = SpecContext::ForVerify(&reqs);

  adapter_->Verify(ctx);

  EXPECT_EQ(req->generated_tokens.size(), 1u);
  EXPECT_EQ(req->generated_tokens[0], 42);
}

// IsCombinableWith: nullptr is always compatible.
TEST_F(PtpDecoderAdapterTest, IsCombinableWithNullptrIsTrue) { EXPECT_TRUE(adapter_->IsCombinableWith(nullptr)); }

// IsCombinableWith: PTP rejects MTP, Trie, and ASR by name (see ptp_decoder_adapter.cpp).
class NamedFakeDecoder : public SpeculativeDecoderInterface {
 public:
  explicit NamedFakeDecoder(const char* name) : name_(name) {}
  const char* GetName() const override { return name_; }
  SpecSchedulingInfo GetSchedulingInfo() const override { return {}; }
  void BeforeForward(SpecContext&) override {}
  void ExecuteForward(SpecContext&) override {}
  void AfterForward(SpecContext&) override {}
  void Verify(SpecContext&) override {}
  bool IsCombinableWith(const SpeculativeDecoderInterface*) const override { return true; }

 private:
  const char* name_;
};

TEST_F(PtpDecoderAdapterTest, IsCombinableWithRejectsMtpTrieAsr) {
  NamedFakeDecoder mtp("MTP");
  NamedFakeDecoder trie("Trie");
  NamedFakeDecoder asr("ASR");
  EXPECT_FALSE(adapter_->IsCombinableWith(&mtp));
  EXPECT_FALSE(adapter_->IsCombinableWith(&trie));
  EXPECT_FALSE(adapter_->IsCombinableWith(&asr));
}

TEST_F(PtpDecoderAdapterTest, IsCombinableWithAcceptsOtherNames) {
  NamedFakeDecoder ptp("PTP");
  NamedFakeDecoder dspark("DSpark");
  EXPECT_TRUE(adapter_->IsCombinableWith(&ptp));
  EXPECT_TRUE(adapter_->IsCombinableWith(&dspark));
}

}  // namespace ksana_llm
