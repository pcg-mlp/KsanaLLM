/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/speculative/asr/asr_decoder_adapter.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "runtime/speculative/asr/asr_custom_generator.h"
#include "runtime/executor_infer_request.h"

namespace ksana_llm {

class AsrDecoderAdapterTest : public testing::Test {
 protected:
  void SetUp() override {
    generator_ = std::make_shared<AsrCustomGenerator>();
    adapter_ = std::make_unique<AsrDecoderAdapter>(generator_);
  }

  std::shared_ptr<AsrCustomGenerator> generator_;
  std::unique_ptr<AsrDecoderAdapter> adapter_;
};

TEST_F(AsrDecoderAdapterTest, GetNameReturnsASR) { EXPECT_STREQ(adapter_->GetName(), "ASR"); }

// ASR does not report extra scheduling resource needs (hard-coded {} in header).
TEST_F(AsrDecoderAdapterTest, GetSchedulingInfoReturnsEmpty) {
  auto info = adapter_->GetSchedulingInfo();
  EXPECT_EQ(info.extra_kv_tokens, 0u);
  EXPECT_EQ(info.extra_token_capacity, 0u);
  EXPECT_EQ(info.workload_multiplier, 0u);
}

// BeforeForward delegates to generator_->GenerateDraft() for every request in ctx.reqs.
// Uses the "full match" path (accepted_tokens covers all draft tokens) which is the
// simplest branch of AsrCustomGenerator::GenerateDraft — it clears draft_tokens.spec.
TEST_F(AsrDecoderAdapterTest, BeforeForwardGeneratesDraftForAllRequests) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 1;
  req->spec_state.draft_tokens.spec = {2, 3};
  req->spec_state.accepted_tokens = {2, 3};
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};

  SpecContext ctx = SpecContext::ForDraftGen(&reqs, nullptr);
  adapter_->BeforeForward(ctx);

  EXPECT_TRUE(req->spec_state.draft_tokens.spec.empty());
}

// BeforeForward is a no-op (does not crash) when ctx.reqs is null.
TEST_F(AsrDecoderAdapterTest, BeforeForwardNoopWhenReqsNull) {
  SpecContext ctx;
  ctx.reqs = nullptr;
  adapter_->BeforeForward(ctx);
  SUCCEED();
}

// ExecuteForward and AfterForward are inline no-ops in the header.
TEST_F(AsrDecoderAdapterTest, ExecuteForwardAndAfterForwardAreNoOps) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 1;
  req->spec_state.draft_tokens.spec = {5, 6};
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};
  SpecContext ctx = SpecContext::ForDraftGen(&reqs, nullptr);

  adapter_->ExecuteForward(ctx);
  adapter_->AfterForward(ctx);

  EXPECT_EQ(req->spec_state.draft_tokens.spec.size(), 2u);
}

// Verify delegates to generator_->FilterDraftTokens(), which (like Trie) is not
// overridden by AsrCustomGenerator and falls back to DefaultFilterDraftTokens.
TEST_F(AsrDecoderAdapterTest, VerifyDelegatesToDefaultFilterDraftTokens) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 1;
  req->spec_state.draft_tokens.spec = {10, 20};
  req->sampling_result_tokens = {10, 20, 30};
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};
  SpecContext ctx = SpecContext::ForVerify(&reqs);

  adapter_->Verify(ctx);

  EXPECT_EQ(req->spec_state.accepted_tokens.size(), 2u);
  EXPECT_EQ(req->generated_tokens[0], 30);
}

TEST_F(AsrDecoderAdapterTest, VerifyNoopWhenReqsNull) {
  SpecContext ctx;
  ctx.reqs = nullptr;
  adapter_->Verify(ctx);
  SUCCEED();
}

// IsCombinableWith: nullptr is always compatible.
TEST_F(AsrDecoderAdapterTest, IsCombinableWithNullptrIsTrue) { EXPECT_TRUE(adapter_->IsCombinableWith(nullptr)); }

class NamedFakeDecoderForAsr : public SpeculativeDecoderInterface {
 public:
  explicit NamedFakeDecoderForAsr(const char* name) : name_(name) {}
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

// ASR is only incompatible with MTP; it accepts PTP/Trie/ASR/DSpark.
TEST_F(AsrDecoderAdapterTest, IsCombinableWithRejectsOnlyMtp) {
  NamedFakeDecoderForAsr mtp("MTP");
  NamedFakeDecoderForAsr ptp("PTP");
  NamedFakeDecoderForAsr dspark("DSpark");
  EXPECT_FALSE(adapter_->IsCombinableWith(&mtp));
  EXPECT_TRUE(adapter_->IsCombinableWith(&ptp));
  EXPECT_TRUE(adapter_->IsCombinableWith(&dspark));
}

}  // namespace ksana_llm
