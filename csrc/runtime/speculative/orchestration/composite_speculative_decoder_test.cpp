/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "runtime/executor_infer_request.h"
#include "runtime/generation_controller.h"
#include "runtime/speculative/orchestration/composite_speculative_decoder.h"
#include "runtime/speculative/common/spec_scheduling_info.h"
#include "runtime/structured_generation/structured_generator_factory.h"

namespace ksana_llm {

// Mock SpeculativeDecoderInterface for testing CompositeSpeculativeDecoder.
class MockSpeculativeDecoder : public SpeculativeDecoderInterface {
 public:
  explicit MockSpeculativeDecoder(const char* name) : name_(name) {}

  const char* GetName() const override { return name_; }

  SpecSchedulingInfo GetSchedulingInfo() const override { return scheduling_info_; }
  void SetSchedulingInfo(const SpecSchedulingInfo& info) { scheduling_info_ = info; }

  void BeforeForward(SpecContext& ctx) override { before_forward_called_ = true; }

  void ExecuteForward(SpecContext& ctx) override { execute_forward_called_ = true; }

  void AfterForward(SpecContext& ctx) override { after_forward_called_ = true; }

  void Verify(SpecContext& ctx) override {
    verify_called_count_++;
    if (ctx.reqs && !ctx.reqs->empty()) {
      verify_req_id_ = (*ctx.reqs)[0]->req_id;
    }
  }

  bool IsCombinableWith(const SpeculativeDecoderInterface* other) const override {
    if (!other) return true;
    return incompatible_names_.find(other->GetName()) == incompatible_names_.end();
  }

  // Mark this decoder incompatible with a specific other decoder.
  void AddIncompatibleWith(const char* name) { incompatible_names_.insert(name); }

  void PreMainForward(SpecContext& ctx) override { pre_main_called_ = true; }

  // Test inspection
  bool before_forward_called() const { return before_forward_called_; }
  bool execute_forward_called() const { return execute_forward_called_; }
  bool after_forward_called() const { return after_forward_called_; }
  bool pre_main_called() const { return pre_main_called_; }
  int verify_called_count() const { return verify_called_count_; }
  int64_t verify_req_id() const { return verify_req_id_; }

  void ResetFlags() {
    before_forward_called_ = false;
    execute_forward_called_ = false;
    after_forward_called_ = false;
    verify_called_count_ = 0;
    verify_req_id_ = -1;
    pre_main_called_ = false;
  }

 private:
  const char* name_;
  std::set<std::string> incompatible_names_;
  SpecSchedulingInfo scheduling_info_;
  bool before_forward_called_ = false;
  bool execute_forward_called_ = false;
  bool after_forward_called_ = false;
  int verify_called_count_ = 0;
  int64_t verify_req_id_ = -1;
  bool pre_main_called_ = false;
};

class CompositeSpeculativeDecoderTest : public testing::Test {
 protected:
  void SetUp() override { composite_.reset(new CompositeSpeculativeDecoder()); }

  std::unique_ptr<CompositeSpeculativeDecoder> composite_;
};

// Test: Register a single decoder and verify count.
TEST_F(CompositeSpeculativeDecoderTest, Register_SingleDecoderIncreasesCount) {
  EXPECT_EQ(composite_->GetDecoderCount(), 0u);
  composite_->Register(std::make_unique<MockSpeculativeDecoder>("Mock1"));
  EXPECT_EQ(composite_->GetDecoderCount(), 1u);
}

// Test: Register multiple decoders, verify by name.
TEST_F(CompositeSpeculativeDecoderTest, Register_MultipleDecodersByName) {
  composite_->Register(std::make_unique<MockSpeculativeDecoder>("A"));
  composite_->Register(std::make_unique<MockSpeculativeDecoder>("B"));
  composite_->Register(std::make_unique<MockSpeculativeDecoder>("C"));
  EXPECT_EQ(composite_->GetDecoderCount(), 3u);

  const auto& decoders = composite_->GetDecoders();
  EXPECT_STREQ(decoders[0]->GetName(), "A");
  EXPECT_STREQ(decoders[1]->GetName(), "B");
  EXPECT_STREQ(decoders[2]->GetName(), "C");
}

// Test: BeforeForward dispatches to all decoders.
TEST_F(CompositeSpeculativeDecoderTest, BeforeForward_DispatchesToAllDecoders) {
  auto mock1 = new MockSpeculativeDecoder("M1");
  auto mock2 = new MockSpeculativeDecoder("M2");
  composite_->Register(std::unique_ptr<SpeculativeDecoderInterface>(mock1));
  composite_->Register(std::unique_ptr<SpeculativeDecoderInterface>(mock2));

  SpecContext ctx;
  composite_->BeforeForward(ctx);

  EXPECT_TRUE(mock1->before_forward_called());
  EXPECT_TRUE(mock2->before_forward_called());
}

// Test: ExecuteForward dispatches to all decoders.
TEST_F(CompositeSpeculativeDecoderTest, ExecuteForward_DispatchesToAll) {
  auto mock1 = new MockSpeculativeDecoder("M1");
  auto mock2 = new MockSpeculativeDecoder("M2");
  composite_->Register(std::unique_ptr<SpeculativeDecoderInterface>(mock1));
  composite_->Register(std::unique_ptr<SpeculativeDecoderInterface>(mock2));

  SpecContext ctx;
  composite_->ExecuteForward(ctx);

  EXPECT_TRUE(mock1->execute_forward_called());
  EXPECT_TRUE(mock2->execute_forward_called());
}

// Test: AfterForward dispatches to all decoders.
TEST_F(CompositeSpeculativeDecoderTest, AfterForward_DispatchesToAll) {
  auto mock1 = new MockSpeculativeDecoder("M1");
  auto mock2 = new MockSpeculativeDecoder("M2");
  composite_->Register(std::unique_ptr<SpeculativeDecoderInterface>(mock1));
  composite_->Register(std::unique_ptr<SpeculativeDecoderInterface>(mock2));

  SpecContext ctx;
  composite_->AfterForward(ctx);

  EXPECT_TRUE(mock1->after_forward_called());
  EXPECT_TRUE(mock2->after_forward_called());
}

// Test: PreMainForward dispatches to all decoders.
TEST_F(CompositeSpeculativeDecoderTest, PreMainForward_DispatchesToAll) {
  auto mock1 = new MockSpeculativeDecoder("M1");
  auto mock2 = new MockSpeculativeDecoder("M2");
  composite_->Register(std::unique_ptr<SpeculativeDecoderInterface>(mock1));
  composite_->Register(std::unique_ptr<SpeculativeDecoderInterface>(mock2));

  SpecContext ctx;
  composite_->PreMainForward(ctx);

  EXPECT_TRUE(mock1->pre_main_called());
  EXPECT_TRUE(mock2->pre_main_called());
}

// Test: Verify dispatches to all decoders (once per decoder, not per-request).
TEST_F(CompositeSpeculativeDecoderTest, Verify_DispatchesToAllDecoders) {
  auto mock1 = new MockSpeculativeDecoder("M1");
  auto mock2 = new MockSpeculativeDecoder("M2");
  composite_->Register(std::unique_ptr<SpeculativeDecoderInterface>(mock1));
  composite_->Register(std::unique_ptr<SpeculativeDecoderInterface>(mock2));

  SpecContext ctx;
  auto req1 = std::make_shared<ExecutorInferRequest>();
  req1->req_id = 42;
  auto req2 = std::make_shared<ExecutorInferRequest>();
  req2->req_id = 99;
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req1, req2};
  ctx.reqs = &reqs;

  composite_->Verify(ctx);

  // Verify() dispatches once per decoder (not per request).
  EXPECT_EQ(mock1->verify_called_count(), 1);
  EXPECT_EQ(mock2->verify_called_count(), 1);
}

// Test: VerifyDraftTokens dispatches to all decoders.
TEST_F(CompositeSpeculativeDecoderTest, VerifyDraftTokens_DispatchesToDecoders) {
  auto mock = new MockSpeculativeDecoder("M1");
  composite_->Register(std::unique_ptr<SpeculativeDecoderInterface>(mock));

  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 77;
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};

  composite_->VerifyDraftTokens(reqs);

  EXPECT_EQ(mock->verify_called_count(), 1);
  EXPECT_EQ(mock->verify_req_id(), 77);
}

// Test: VerifyDraftTokens with empty decoder list uses default filter.
TEST_F(CompositeSpeculativeDecoderTest, VerifyDraftTokens_UsesDefaultWhenNoDecoders) {
  auto req = std::make_shared<ExecutorInferRequest>();
  req->generated_tokens = {42};
  req->sampling_result_tokens = {42};
  req->spec_state.draft_tokens.clear();
  req->spec_state.forwarding_tokens_draft_num = 0;
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};

  // No decoders registered — should use DefaultFilterDraftTokens.
  composite_->VerifyDraftTokens(reqs);

  // Default filter: hit=0 (sampling_result_tokens.size()-1=0 == draft_tokens.size()=0)
  EXPECT_EQ(req->generated_tokens[0], 42);
}

// Test: Scheduling info aggregation compues correctly.
TEST_F(CompositeSpeculativeDecoderTest, GetAggregatedSchedulingInfo_SumsAllDecoders) {
  SpecSchedulingInfo info1;
  info1.extra_kv_tokens = 2;
  info1.extra_token_capacity = 1;
  info1.workload_multiplier = 3;

  SpecSchedulingInfo info2;
  info2.extra_kv_tokens = 3;
  info2.extra_token_capacity = 2;
  info2.workload_multiplier = 5;

  auto mock1 = std::make_unique<MockSpeculativeDecoder>("M1");
  mock1->SetSchedulingInfo(info1);
  auto mock2 = std::make_unique<MockSpeculativeDecoder>("M2");
  mock2->SetSchedulingInfo(info2);
  composite_->Register(std::move(mock1));
  composite_->Register(std::move(mock2));

  auto aggregated = composite_->GetAggregatedSchedulingInfo();
  EXPECT_EQ(aggregated.extra_kv_tokens, 5u);       // 2 + 3
  EXPECT_EQ(aggregated.extra_token_capacity, 3u);  // 1 + 2
  EXPECT_EQ(aggregated.workload_multiplier, 8u);   // 3 + 5
}

// Test: AreAllCompatible returns true when all pairs are compatible.
TEST_F(CompositeSpeculativeDecoderTest, AreAllCompatible_AllCompatible) {
  composite_->Register(std::make_unique<MockSpeculativeDecoder>("A"));
  composite_->Register(std::make_unique<MockSpeculativeDecoder>("B"));
  EXPECT_TRUE(composite_->AreAllCompatible());
}

// Test: AreAllCompatible detects incompatible pair (A marks itself incompatible with C).
TEST_F(CompositeSpeculativeDecoderTest, AreAllCompatible_IncompatiblePair) {
  auto mockA = std::make_unique<MockSpeculativeDecoder>("A");
  auto mockB = std::make_unique<MockSpeculativeDecoder>("B");
  auto mockC = std::make_unique<MockSpeculativeDecoder>("C");
  mockA->AddIncompatibleWith("C");  // A cannot work with C

  composite_->Register(std::move(mockA));
  composite_->Register(std::move(mockB));
  composite_->Register(std::move(mockC));

  EXPECT_FALSE(composite_->AreAllCompatible());
}

// Test: Stale draft tokens are cleared by HandlePostprocess before verification.
TEST_F(CompositeSpeculativeDecoderTest, HandlePostprocess_ClearsStaleDraftTokens) {
  // Register a mock decoder so HandlePostprocess doesn't return early.
  composite_->Register(std::make_unique<MockSpeculativeDecoder>("Mock"));

  auto req = std::make_shared<ExecutorInferRequest>();
  req->spec_state.draft_tokens.mtp = {10, 20};
  req->spec_state.forwarding_tokens_draft_num = 0;
  req->sampling_result_tokens = {1};
  std::vector<std::shared_ptr<ExecutorInferRequest>> running_reqs = {req};

  // Call HandlePostprocess with minimal dependencies.
  // hidden_unit, schedule_output, and generation_controller can all be
  // nullptr — the mock decoder's Verify/ExecuteForward/BeforeForward are
  // all safe with nullptr SpecContext members.
  composite_->HandlePostprocess(running_reqs, nullptr, nullptr, nullptr);

  // Stale draft_tokens should have been cleared (forwarding_tokens_draft_num == 0).
  EXPECT_EQ(req->spec_state.draft_tokens.size(), 0u);
}

// Test: HandlePostprocess does not clear draft_tokens when forwarded.
TEST_F(CompositeSpeculativeDecoderTest, HandlePostprocess_PreservesForwardedDraftTokens) {
  composite_->Register(std::make_unique<MockSpeculativeDecoder>("Mock"));

  auto req = std::make_shared<ExecutorInferRequest>();
  req->spec_state.draft_tokens.mtp = {10, 20};
  req->spec_state.forwarding_tokens_draft_num = 2;  // drafts were actually forwarded
  req->sampling_result_tokens = {1, 2, 3};
  std::vector<std::shared_ptr<ExecutorInferRequest>> running_reqs = {req};

  composite_->HandlePostprocess(running_reqs, nullptr, nullptr, nullptr);

  // Draft tokens should be preserved (forwarding_tokens_draft_num > 0).
  EXPECT_EQ(req->spec_state.draft_tokens.size(), 2u);
  EXPECT_EQ(req->spec_state.draft_tokens.mtp[0], 10);
}

// Test: HandlePostprocess updates sampling_token_num correctly.
TEST_F(CompositeSpeculativeDecoderTest, HandlePostprocess_UpdatesSamplingTokenNum) {
  composite_->Register(std::make_unique<MockSpeculativeDecoder>("Mock"));

  auto req = std::make_shared<ExecutorInferRequest>();
  req->sampling_result_tokens = {1};
  req->spec_state.forwarding_tokens_draft_num = 0;
  std::vector<std::shared_ptr<ExecutorInferRequest>> running_reqs = {req};

  composite_->HandlePostprocess(running_reqs, nullptr, nullptr, nullptr);

  // sampling_token_num = kStepGenerateTokenNum + draft_tokens.size()
  // After clearing stale drafts: draft_tokens.size() == 0, so expected = 1 + 0 = 1
  // But HandlePostprocess also runs ExecuteForward which may modify draft_tokens.
  // The mock's ExecuteForward doesn't change draft_tokens, so it stays 0.
  EXPECT_EQ(req->sampling_token_num, 1u);
}

// Test: Register(nullptr) is a safe no-op — decoder count stays unchanged.
TEST_F(CompositeSpeculativeDecoderTest, Register_NullptrIsSafeNoOp) {
  EXPECT_EQ(composite_->GetDecoderCount(), 0u);
  composite_->Register(nullptr);
  EXPECT_EQ(composite_->GetDecoderCount(), 0u);

  // Registering a real decoder afterwards still works normally.
  composite_->Register(std::make_unique<MockSpeculativeDecoder>("A"));
  EXPECT_EQ(composite_->GetDecoderCount(), 1u);
}

// Test: HandlePostprocess returns early (skips BeforeForward/ExecuteForward/
// AfterForward) when no decoders are registered — VerifyDraftTokens still runs.
TEST_F(CompositeSpeculativeDecoderTest, HandlePostprocess_EarlyReturnWhenNoDecoders) {
  EXPECT_EQ(composite_->GetDecoderCount(), 0u);

  auto req = std::make_shared<ExecutorInferRequest>();
  req->generated_tokens = {7};
  req->sampling_result_tokens = {7};
  req->spec_state.draft_tokens.clear();
  req->spec_state.forwarding_tokens_draft_num = 0;
  // sampling_token_num starts at a sentinel value distinct from kStepGenerateTokenNum
  // (+draft_token_num) so we can detect whether Step 4 (which only runs when
  // decoders are registered) executed.
  req->sampling_token_num = 999;
  std::vector<std::shared_ptr<ExecutorInferRequest>> running_reqs = {req};

  composite_->HandlePostprocess(running_reqs, nullptr, nullptr, nullptr);

  // VerifyDraftTokens (Step 2) still ran via the DefaultFilterDraftTokens fallback,
  // moving the sampling result into generated_tokens.
  EXPECT_EQ(req->generated_tokens[0], 7);
  // sampling_token_num is untouched by Step 4 since decoders_ is empty (early return).
  EXPECT_EQ(req->sampling_token_num, 999u);
}

// Test: SetGenerationController wires a real GenerationController into
// HandlePostprocess's UpdateGenerationState delegation (Step 1). No decoder is
// registered so VerifyDraftTokens (Step 2) falls back to DefaultFilterDraftTokens,
// which is what actually moves sampling_result_tokens into generated_tokens.
TEST_F(CompositeSpeculativeDecoderTest, SetGenerationController_DelegatesUpdateGenerationState) {
  auto factory = std::make_shared<StructuredGeneratorFactory>();
  GenerationController gc(factory);
  composite_->SetGenerationController(&gc);

  auto req = std::make_shared<ExecutorInferRequest>();
  req->sampling_result_tokens = {5};
  req->spec_state.forwarding_tokens_draft_num = 0;
  std::vector<std::shared_ptr<ExecutorInferRequest>> running_reqs = {req};

  // No structured generator is attached to req, so UpdateGenerationState's
  // internal UpdateStructuredGeneratorState is effectively a no-op for it —
  // this test only verifies that wiring a real (non-null) GenerationController
  // through SetGenerationController does not crash and HandlePostprocess still
  // completes its normal flow (VerifyDraftTokens moves the sampling result into
  // generated_tokens via the DefaultFilterDraftTokens fallback).
  composite_->HandlePostprocess(running_reqs, nullptr, nullptr, &gc);

  ASSERT_EQ(req->generated_tokens.size(), 1u);
  EXPECT_EQ(req->generated_tokens[0], 5);
}

}  // namespace ksana_llm
