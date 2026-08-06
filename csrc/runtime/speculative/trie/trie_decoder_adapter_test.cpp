/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/speculative/trie/trie_decoder_adapter.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "runtime/speculative/trie/trie_generator.h"
#include "runtime/executor_infer_request.h"

namespace ksana_llm {

class TrieDecoderAdapterTest : public testing::Test {
 protected:
  void SetUp() override {
    generator_ = std::make_shared<TrieGenerator>();
    adapter_ = std::make_unique<TrieDecoderAdapter>(generator_);
  }

  void TearDown() override { generator_->GetTrie()->Reset(); }

  std::shared_ptr<TrieGenerator> generator_;
  std::unique_ptr<TrieDecoderAdapter> adapter_;
};

TEST_F(TrieDecoderAdapterTest, GetNameReturnsTrie) { EXPECT_STREQ(adapter_->GetName(), "Trie"); }

// Trie does not report extra scheduling resource needs (hard-coded {} in header).
TEST_F(TrieDecoderAdapterTest, GetSchedulingInfoReturnsEmpty) {
  auto info = adapter_->GetSchedulingInfo();
  EXPECT_EQ(info.extra_kv_tokens, 0u);
  EXPECT_EQ(info.extra_token_capacity, 0u);
  EXPECT_EQ(info.workload_multiplier, 0u);
}

// BeforeForward delegates to generator_->GenerateDraft() for every request in ctx.reqs.
TEST_F(TrieDecoderAdapterTest, BeforeForwardGeneratesDraftForAllRequests) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 1;
  req->step = 1;
  req->spec_state.forwarding_tokens_draft_num = 0;
  req->forwarding_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};

  SpecContext ctx = SpecContext::ForDraftGen(&reqs, nullptr);
  adapter_->BeforeForward(ctx);

  // GenerateDraft populates req->spec_state.draft_tokens.spec (may be empty on a cold
  // trie, but the call itself must not crash and must complete normally).
  SUCCEED();
}

// BeforeForward is a no-op (does not crash) when ctx.reqs is null.
TEST_F(TrieDecoderAdapterTest, BeforeForwardNoopWhenReqsNull) {
  SpecContext ctx;
  ctx.reqs = nullptr;
  adapter_->BeforeForward(ctx);
  SUCCEED();
}

// ExecuteForward and AfterForward are inline no-ops in the header.
TEST_F(TrieDecoderAdapterTest, ExecuteForwardAndAfterForwardAreNoOps) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 1;
  req->spec_state.draft_tokens.spec = {5, 6};
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};
  SpecContext ctx = SpecContext::ForDraftGen(&reqs, nullptr);

  adapter_->ExecuteForward(ctx);
  adapter_->AfterForward(ctx);

  EXPECT_EQ(req->spec_state.draft_tokens.spec.size(), 2u);
}

// Verify delegates to generator_->FilterDraftTokens(), which (unlike PTP) is not
// overridden by TrieGenerator and falls back to DefaultFilterDraftTokens.
TEST_F(TrieDecoderAdapterTest, VerifyDelegatesToDefaultFilterDraftTokens) {
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

TEST_F(TrieDecoderAdapterTest, VerifyNoopWhenReqsNull) {
  SpecContext ctx;
  ctx.reqs = nullptr;
  adapter_->Verify(ctx);
  SUCCEED();
}

// IsCombinableWith: nullptr is always compatible.
TEST_F(TrieDecoderAdapterTest, IsCombinableWithNullptrIsTrue) { EXPECT_TRUE(adapter_->IsCombinableWith(nullptr)); }

class NamedFakeDecoderForTrie : public SpeculativeDecoderInterface {
 public:
  explicit NamedFakeDecoderForTrie(const char* name) : name_(name) {}
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

// Trie is only incompatible with MTP; it accepts PTP/Trie/ASR/DSpark.
TEST_F(TrieDecoderAdapterTest, IsCombinableWithRejectsOnlyMtp) {
  NamedFakeDecoderForTrie mtp("MTP");
  NamedFakeDecoderForTrie ptp("PTP");
  NamedFakeDecoderForTrie dspark("DSpark");
  EXPECT_FALSE(adapter_->IsCombinableWith(&mtp));
  EXPECT_TRUE(adapter_->IsCombinableWith(&ptp));
  EXPECT_TRUE(adapter_->IsCombinableWith(&dspark));
}

}  // namespace ksana_llm
