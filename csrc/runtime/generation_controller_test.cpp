/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "runtime/generation_controller.h"
#include "runtime/speculative/common/draft_generator_interface.h"
#include "runtime/structured_generation/reasoning_structured_generator.h"

namespace ksana_llm {

// Mock StructuredGeneratorInterface for testing
class MockStructuredGenerator : public StructuredGeneratorInterface {
 public:
  explicit MockStructuredGenerator(const std::vector<int>& expected_output_tokens)
      : expected_output_tokens_(expected_output_tokens) {}

  bool AcceptToken(int token_id) override {
    if (expecting_idx_ >= expected_output_tokens_.size()) {
      return false;
    }
    // 只能顺序接收 expected_output_tokens_ 列表中的 token
    if (expected_output_tokens_[expecting_idx_] == token_id) {
      KLLM_LOG_DEBUG << "Accept token: " << token_id << ", expecting idx: " << expecting_idx_;
      expecting_idx_++;
      return true;
    }
    return false;
  }

  bool FillNextTokenBitmask(void* next_token_bitmask) override { return true; }

  void Rollback(int rollback_token_num) override {
    KLLM_CHECK(static_cast<size_t>(rollback_token_num) <= expecting_idx_);
    expecting_idx_ -= rollback_token_num;
  }

  void Reset() override { expecting_idx_ = 0; }

  bool FindJumpForwardTokens(std::vector<int>& jump_tokens) override { return false; }

  bool IsTerminated() const override { return false; }

  bool IsCompleted() const override { return expecting_idx_ == expected_output_tokens_.size(); }

  bool IsValid() const override { return true; }

  StructuredConstraintType GetConstraintType() const override { return StructuredConstraintType::REGEX; }

 private:
  std::vector<int> expected_output_tokens_;
  size_t expecting_idx_ = 0;
};

class MockGeneratorCreator : public GeneratorCreator {
 public:
  explicit MockGeneratorCreator(const std::vector<int>& expected_output_tokens)
      : expected_output_tokens_(expected_output_tokens) {}
  std::shared_ptr<StructuredGeneratorInterface> CreateGenerator(const StructuredGeneratorConfig& config) override {
    return std::make_shared<MockStructuredGenerator>(expected_output_tokens_);
  }

  StructuredConstraintType GetConstraintType() const override { return StructuredConstraintType::REGEX; }

 private:
  std::vector<int> expected_output_tokens_;
};

class GenerationControllerTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    std::shared_ptr<StructuredGeneratorFactory> factory = std::make_shared<StructuredGeneratorFactory>();
    std::vector<int> expected_output_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    factory->RegisterCreator(StructuredConstraintType::REGEX,
                             std::make_unique<MockGeneratorCreator>(expected_output_tokens));
    generation_controller_ = std::make_shared<GenerationController>(factory);
  }

  std::shared_ptr<Request> CreateMockRequest(const std::vector<int>& stop_token_ids = {}) {
    auto python_input = std::make_shared<KsanaPythonInput>();
    python_input->model_name = "test_model";
    python_input->input_tokens = {1, 2, 3};
    python_input->sampling_config.stop_token_ids = stop_token_ids;

    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    auto request = std::make_shared<Request>(python_input, req_ctx);

    request->req_id = 1;
    request->model_name = "test_model";
    request->input_tokens = {1, 2, 3};
    request->output_tokens = {};
    request->logprobs = {};
    request->sampling_config.stop_token_ids = stop_token_ids;
    request->finished = false;
    request->aborted = false;

    return request;
  }

  std::shared_ptr<ExecutorInferRequest> CreateMockExecutorInferRequest(int64_t req_id) {
    auto exec_req = std::make_shared<ExecutorInferRequest>();
    exec_req->req_id = req_id;
    exec_req->input_tokens = {1, 2, 3};
    exec_req->output_tokens = {};
    exec_req->finished = false;
    exec_req->aborted = false;
    return exec_req;
  }

  std::shared_ptr<GenerationController> generation_controller_;
};

// Test CreateGenerators method
TEST_F(GenerationControllerTest, CreateGenerators) {
  auto schedule_output = std::make_shared<ScheduleOutput>();

  // Create a mock executor request with structured generator config
  auto exec_req = CreateMockExecutorInferRequest(1);
  exec_req->structured_generator_config.constraint_type = StructuredConstraintType::REGEX;
  exec_req->structured_generator_config.constraint_spec = "dummy";

  schedule_output->executor_running_reqs.push_back(exec_req);
  schedule_output->need_create_generator_req_ids.insert(1);

  generation_controller_->CreateGenerators(schedule_output);

  // Verify generator was created and assigned
  EXPECT_TRUE(exec_req->structured_generator != nullptr);
  EXPECT_TRUE(exec_req->structured_generator->IsValid());
}

TEST_F(GenerationControllerTest, UpdateGenerationState) {
  // 测试 structured_generator 拒绝新生成 token 的情况

  std::vector<int> expected_output_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  auto structured_generator = std::make_shared<MockStructuredGenerator>(expected_output_tokens);
  auto exec_req = CreateMockExecutorInferRequest(1);
  exec_req->structured_generator = structured_generator;
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {exec_req};

  // First sampling_result_tokens is acceptable, generated_tokens[0] is sampling_result_tokens[0];
  exec_req->sampling_result_tokens = {1};
  exec_req->spec_state.draft_tokens.clear();
  generation_controller_->UpdateGenerationState(reqs);
  // After refactor, verification moves to composite decoder; simulate HandlePostprocess order:
  // UpdateGenerationState (clears stale draft_tokens) → DefaultFilterDraftTokens (verify).
  DefaultFilterDraftTokens(exec_req);
  EXPECT_EQ(exec_req->spec_state.accepted_tokens.size(), 0);
  EXPECT_EQ(exec_req->generated_tokens[0], 1);

  // Expecting 2 to be generated, but sampling result is 1, so generated_tokens[0] is 1 (was -1 in old test,
  // but DefaultFilterDraftTokens does not produce -1 — TODO(robertyuan): scheduler doesn't handle this value).
  exec_req->sampling_result_tokens = {1};
  exec_req->spec_state.draft_tokens.clear();
  generation_controller_->UpdateGenerationState(reqs);
  DefaultFilterDraftTokens(exec_req);
  EXPECT_EQ(exec_req->spec_state.accepted_tokens.size(), 0);

  // Expecting 2,3 to be generated, draft_tokens is 2 and sampling result is 2,3, so accepted_token is 2,
  // generated_tokens[0] is 3;
  exec_req->sampling_result_tokens = {2, 3};
  exec_req->spec_state.draft_tokens.mtp = {2};
  exec_req->spec_state.forwarding_tokens_draft_num = 1;
  generation_controller_->UpdateGenerationState(reqs);
  DefaultFilterDraftTokens(exec_req);
  EXPECT_EQ(exec_req->spec_state.accepted_tokens.size(), 1);
  EXPECT_EQ(exec_req->spec_state.accepted_tokens[0], 2);
  EXPECT_EQ(exec_req->generated_tokens[0], 3);

  // Expecting 4,5,6 to be generated, draft_tokens is 4,5 and sampling result is 4,7,8, so accepted_token is null
  // because generated token 7 is rejected by structure generator, generated_tokens[0] is 4;
  // computing for draft token 4 is wasted.
  exec_req->sampling_result_tokens = {4, 7, 8};
  exec_req->spec_state.draft_tokens.mtp = {4, 5};
  exec_req->spec_state.forwarding_tokens_draft_num = 2;
  generation_controller_->UpdateGenerationState(reqs);
  DefaultFilterDraftTokens(exec_req);
  EXPECT_EQ(exec_req->spec_state.accepted_tokens.size(), 0);
  EXPECT_EQ(exec_req->generated_tokens[0], 4);

  // Expecting 5,6,7 to be generated, draft_tokens is 5,6 and sampling result is 5,6,7, so accepted_token is 5,6 because
  // 6,7 are accepted by structure generator, generated_tokens[0] is 7;
  exec_req->sampling_result_tokens = {5, 6, 7};
  exec_req->spec_state.draft_tokens.mtp = {5, 6};
  exec_req->spec_state.forwarding_tokens_draft_num = 2;
  generation_controller_->UpdateGenerationState(reqs);
  DefaultFilterDraftTokens(exec_req);
  EXPECT_EQ(exec_req->spec_state.accepted_tokens.size(), 2);
  EXPECT_EQ(exec_req->spec_state.accepted_tokens[0], 5);
  EXPECT_EQ(exec_req->spec_state.accepted_tokens[1], 6);
  EXPECT_EQ(exec_req->generated_tokens[0], 7);
}

// 测试 ReasoningStructuredGenerator - 有思考内容的模型
// 当 think_end_token_id > 0 时，会包装 ReasoningStructuredGenerator
TEST_F(GenerationControllerTest, ReasoningStructuredGeneratorWithReasoning) {
  // 期望的输出序列: [10, 20, 30]
  std::vector<int> expected_output_tokens = {10, 20, 30};
  auto inner_generator = std::make_shared<MockStructuredGenerator>(expected_output_tokens);

  // 思考结束标记为 999（> 0，表示支持思考）
  int think_end_token_id = 999;
  auto reasoner_generator = std::make_shared<ReasoningStructuredGenerator>(inner_generator, think_end_token_id, true);

  // 验证初始状态：处于推理阶段
  EXPECT_TRUE(reasoner_generator->IsInReasoningPhase());
  EXPECT_FALSE(reasoner_generator->IsTerminated());
  EXPECT_TRUE(reasoner_generator->IsValid());

  // 在推理阶段，任何 token 都应该被接受（模拟思考内容）
  EXPECT_TRUE(reasoner_generator->AcceptToken(1));
  EXPECT_TRUE(reasoner_generator->AcceptToken(2));
  EXPECT_TRUE(reasoner_generator->AcceptToken(3));
  EXPECT_TRUE(reasoner_generator->IsInReasoningPhase());

  // 遇到思考结束标记，退出推理阶段
  EXPECT_TRUE(reasoner_generator->AcceptToken(think_end_token_id));
  EXPECT_FALSE(reasoner_generator->IsInReasoningPhase());

  // 推理阶段结束后，只接受符合约束的 token
  EXPECT_TRUE(reasoner_generator->AcceptToken(10));   // 第一个期望的 token
  EXPECT_FALSE(reasoner_generator->AcceptToken(15));  // 不符合期望的 token
  EXPECT_TRUE(reasoner_generator->AcceptToken(20));   // 第二个期望的 token
  EXPECT_TRUE(reasoner_generator->AcceptToken(30));   // 第三个期望的 token
  EXPECT_TRUE(reasoner_generator->IsCompleted());
  EXPECT_FALSE(reasoner_generator->IsTerminated());

  // 验证约束类型
  EXPECT_EQ(reasoner_generator->GetConstraintType(), StructuredConstraintType::REGEX);
}

TEST_F(GenerationControllerTest, ReasoningStructuredGeneratorReset) {
  auto inner_generator = std::make_shared<MockStructuredGenerator>(std::vector<int>{10});
  auto reasoner_generator = std::make_shared<ReasoningStructuredGenerator>(inner_generator, 999, true);

  EXPECT_FALSE(reasoner_generator->IsCompleted());
  EXPECT_TRUE(reasoner_generator->AcceptToken(1));
  EXPECT_TRUE(reasoner_generator->AcceptToken(999));
  EXPECT_TRUE(reasoner_generator->AcceptToken(10));
  EXPECT_TRUE(reasoner_generator->IsCompleted());

  reasoner_generator->Reset();
  EXPECT_TRUE(reasoner_generator->IsInReasoningPhase());
  EXPECT_FALSE(reasoner_generator->IsCompleted());
  EXPECT_FALSE(reasoner_generator->IsTerminated());

  EXPECT_TRUE(reasoner_generator->AcceptToken(999));
  EXPECT_TRUE(reasoner_generator->AcceptToken(10));
  EXPECT_TRUE(reasoner_generator->IsCompleted());
}

TEST_F(GenerationControllerTest, ReasoningStructuredGeneratorResetPreservesDisabledReasoning) {
  auto inner_generator = std::make_shared<MockStructuredGenerator>(std::vector<int>{10});
  auto reasoner_generator = std::make_shared<ReasoningStructuredGenerator>(inner_generator, 999, false);

  EXPECT_TRUE(reasoner_generator->AcceptToken(10));
  EXPECT_TRUE(reasoner_generator->IsCompleted());
  reasoner_generator->Reset();

  EXPECT_FALSE(reasoner_generator->IsInReasoningPhase());
  EXPECT_FALSE(reasoner_generator->IsCompleted());
  EXPECT_TRUE(reasoner_generator->AcceptToken(10));
}

// 测试没有思考内容的模型 - 直接使用Xgrammar 作为约束解码后端
// 这个测试模拟 StructuredGeneratorFactory 在 think_end_token_id <= 0 时的行为
TEST_F(GenerationControllerTest, StructuredGeneratorWithoutReasoning) {
  // 期望的输出序列: [100, 200, 300]
  std::vector<int> expected_output_tokens = {100, 200, 300};

  // 对于不支持思考的模型，直接使用内部生成器，不包装 ReasoningStructuredGenerator
  auto generator = std::make_shared<MockStructuredGenerator>(expected_output_tokens);

  // 验证生成器有效
  EXPECT_TRUE(generator->IsValid());
  EXPECT_TRUE(generator->AcceptToken(100));   // 第一个期望的 token
  EXPECT_FALSE(generator->AcceptToken(150));  // 不符合期望的 token，被拒绝
  EXPECT_TRUE(generator->AcceptToken(200));   // 第二个期望的 token
  EXPECT_FALSE(generator->AcceptToken(250));  // 不符合期望的 token，被拒绝
  EXPECT_TRUE(generator->AcceptToken(300));   // 第三个期望的 token

  // 验证生成器仍然有效
  EXPECT_TRUE(generator->IsValid());
  EXPECT_EQ(generator->GetConstraintType(), StructuredConstraintType::REGEX);
}

// 测试 FillNextTokenBitmask - 推理阶段不应用约束
TEST_F(GenerationControllerTest, ReasoningStructuredGeneratorFillNextTokenBitmask) {
  std::vector<int> expected_output_tokens = {10, 20, 30};
  auto inner_generator = std::make_shared<MockStructuredGenerator>(expected_output_tokens);

  int think_end_token_id = 999;
  auto reasoner_generator = std::make_shared<ReasoningStructuredGenerator>(inner_generator, think_end_token_id, true);

  void* dummy_bitmask = nullptr;

  // 在推理阶段，FillNextTokenBitmask 应该返回 false（不应用约束）
  EXPECT_TRUE(reasoner_generator->IsInReasoningPhase());
  EXPECT_FALSE(reasoner_generator->FillNextTokenBitmask(dummy_bitmask));

  // 退出推理阶段
  reasoner_generator->AcceptToken(think_end_token_id);
  EXPECT_FALSE(reasoner_generator->IsInReasoningPhase());

  // 推理阶段结束后，FillNextTokenBitmask 应该返回 true（应用约束）
  EXPECT_TRUE(reasoner_generator->FillNextTokenBitmask(dummy_bitmask));
}

// 测试 Rollback - 推理阶段不传递回滚
TEST_F(GenerationControllerTest, ReasoningStructuredGeneratorRollback) {
  std::vector<int> expected_output_tokens = {10, 20, 30};
  auto inner_generator = std::make_shared<MockStructuredGenerator>(expected_output_tokens);

  int think_end_token_id = 999;
  auto reasoner_generator = std::make_shared<ReasoningStructuredGenerator>(inner_generator, think_end_token_id, true);

  // 在推理阶段接受一些 token
  reasoner_generator->AcceptToken(1);
  reasoner_generator->AcceptToken(2);
  reasoner_generator->AcceptToken(3);

  // 在推理阶段回滚不会影响内部生成器（不会抛出异常）
  EXPECT_NO_THROW(reasoner_generator->Rollback(2));
  EXPECT_TRUE(reasoner_generator->IsInReasoningPhase());

  // 退出推理阶段
  reasoner_generator->AcceptToken(think_end_token_id);
  EXPECT_FALSE(reasoner_generator->IsInReasoningPhase());

  // 推理阶段结束后接受一些 token
  reasoner_generator->AcceptToken(10);
  reasoner_generator->AcceptToken(20);

  // 推理阶段结束后，回滚会传递给内部生成器
  EXPECT_NO_THROW(reasoner_generator->Rollback(1));

  // 验证回滚后可以重新接受 token
  EXPECT_TRUE(reasoner_generator->AcceptToken(20));

  // 测试负数回滚应该抛出异常
  EXPECT_THROW(reasoner_generator->Rollback(-1), std::invalid_argument);
}

TEST_F(GenerationControllerTest, ReasoningStructuredGeneratorRollbackAcrossBoundary) {
  auto inner_generator = std::make_shared<MockStructuredGenerator>(std::vector<int>{10, 20});
  auto reasoner_generator = std::make_shared<ReasoningStructuredGenerator>(inner_generator, 999, true);

  EXPECT_TRUE(reasoner_generator->AcceptToken(1));
  EXPECT_TRUE(reasoner_generator->AcceptToken(2));
  EXPECT_TRUE(reasoner_generator->AcceptToken(999));
  EXPECT_TRUE(reasoner_generator->AcceptToken(10));
  EXPECT_TRUE(reasoner_generator->AcceptToken(20));

  // Roll back both constrained tokens and the reasoning boundary itself.
  EXPECT_NO_THROW(reasoner_generator->Rollback(3));
  EXPECT_TRUE(reasoner_generator->IsInReasoningPhase());

  EXPECT_TRUE(reasoner_generator->AcceptToken(999));
  EXPECT_TRUE(reasoner_generator->AcceptToken(10));
  EXPECT_TRUE(reasoner_generator->AcceptToken(20));
}

TEST_F(GenerationControllerTest, ThinkEndTokenIsConstrainedWhenReasoningIsDisabled) {
  auto inner_generator = std::make_shared<MockStructuredGenerator>(std::vector<int>{999});
  auto reasoner_generator = std::make_shared<ReasoningStructuredGenerator>(inner_generator, 999, false);

  EXPECT_FALSE(reasoner_generator->IsInReasoningPhase());
  EXPECT_TRUE(reasoner_generator->AcceptToken(999));
  EXPECT_FALSE(reasoner_generator->IsInReasoningPhase());
  EXPECT_NO_THROW(reasoner_generator->Rollback(1));
  EXPECT_TRUE(reasoner_generator->AcceptToken(999));
}

TEST_F(GenerationControllerTest, StructuralTagBypassesLegacyReasoningWrapper) {
  VocabInfo vocab_info;
  vocab_info.think_end_token_id = 999;
  auto factory = std::make_shared<StructuredGeneratorFactory>(vocab_info);
  const std::vector<int> expected_output_tokens = {1};
  factory->RegisterCreator(StructuredConstraintType::REGEX,
                           std::make_unique<MockGeneratorCreator>(expected_output_tokens));
  factory->RegisterCreator(StructuredConstraintType::STRUCTURAL_TAG,
                           std::make_unique<MockGeneratorCreator>(expected_output_tokens));

  auto regex_generator = factory->CreateGenerator(
      StructuredGeneratorConfig(StructuredConstraintType::REGEX, "[0-9]+"), true);
  EXPECT_NE(std::dynamic_pointer_cast<ReasoningStructuredGenerator>(regex_generator), nullptr);

  auto structural_tag_generator = factory->CreateGenerator(
      StructuredGeneratorConfig(StructuredConstraintType::STRUCTURAL_TAG,
                                R"({"type":"structural_tag","format":{"type":"const_string","value":"1"}})"),
      true);
  EXPECT_EQ(std::dynamic_pointer_cast<ReasoningStructuredGenerator>(structural_tag_generator), nullptr);
}

// 测试 FindJumpForwardTokens - 推理阶段不支持跳跃前进
TEST_F(GenerationControllerTest, ReasoningStructuredGeneratorFindJumpForwardTokens) {
  std::vector<int> expected_output_tokens = {10, 20, 30};
  auto inner_generator = std::make_shared<MockStructuredGenerator>(expected_output_tokens);

  int think_end_token_id = 999;
  auto reasoner_generator = std::make_shared<ReasoningStructuredGenerator>(inner_generator, think_end_token_id, true);

  std::vector<int> jump_tokens;

  // 在推理阶段，FindJumpForwardTokens 应该返回 false 并清空 jump_tokens
  EXPECT_TRUE(reasoner_generator->IsInReasoningPhase());
  jump_tokens = {1, 2, 3};
  EXPECT_FALSE(reasoner_generator->FindJumpForwardTokens(jump_tokens));
  EXPECT_TRUE(jump_tokens.empty());

  // 退出推理阶段
  reasoner_generator->AcceptToken(think_end_token_id);
  EXPECT_FALSE(reasoner_generator->IsInReasoningPhase());

  // 推理阶段结束后，FindJumpForwardTokens 会传递给内部生成器
  // MockStructuredGenerator 的实现返回 false
  EXPECT_FALSE(reasoner_generator->FindJumpForwardTokens(jump_tokens));
}

// After refactor, GenerationController::UpdateGenerationState no longer
// clears stale draft_tokens — that responsibility has moved to
// CompositeSpeculativeDecoder::HandlePostprocess.
// Test that UpdateGenerationState leaves draft_tokens untouched.
TEST_F(GenerationControllerTest, UpdateGenerationStateDoesNotTouchDraftTokens) {
  auto exec_req = CreateMockExecutorInferRequest(1);
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {exec_req};

  exec_req->spec_state.draft_tokens.mtp = {10, 11};
  exec_req->spec_state.forwarding_tokens_draft_num = 0;
  exec_req->sampling_result_tokens = {1};

  generation_controller_->UpdateGenerationState(reqs);

  // draft_tokens should NOT be cleared by UpdateGenerationState anymore —
  // that cleanup is now done in CompositeSpeculativeDecoder::HandlePostprocess
  // before VerifyDraftTokens.
  EXPECT_EQ(exec_req->spec_state.draft_tokens.size(), 2);
  EXPECT_EQ(exec_req->spec_state.draft_tokens.mtp[0], 10);
}

// After refactor: UpdateGenerationState no longer handles draft_tokens at all,
// regardless of forwarding_tokens_draft_num. Test that draft_tokens are always
// left intact by UpdateGenerationState.
TEST_F(GenerationControllerTest, UpdateGenerationStateAlwaysPreservesDraftTokens) {
  auto exec_req = CreateMockExecutorInferRequest(1);
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {exec_req};

  // draft_tokens present with forwarding active.
  exec_req->spec_state.draft_tokens.mtp = {1};
  exec_req->spec_state.forwarding_tokens_draft_num = 1;
  exec_req->sampling_result_tokens = {1, 2};

  generation_controller_->UpdateGenerationState(reqs);

  // draft_tokens preserved (cleanup moved to HandlePostprocess).
  EXPECT_EQ(exec_req->spec_state.draft_tokens.size(), 1);

  // After refactor, verification moves to composite decoder; simulate HandlePostprocess order.
  DefaultFilterDraftTokens(exec_req);
  EXPECT_EQ(exec_req->generated_tokens[0], 2);
  EXPECT_EQ(exec_req->spec_state.accepted_tokens[0], 1);
}

// After refactor, stale draft_tokens clearing is done by
// CompositeSpeculativeDecoder::HandlePostprocess (right after UpdateGenerationState,
// before VerifyDraftTokens). UpdateGenerationState itself no longer clears draft_tokens.
// Simulate the HandlePostprocess order: clear stale drafts → verify.
TEST_F(GenerationControllerTest, VerifyAfterDraftCleanupOrder) {
  auto exec_req = CreateMockExecutorInferRequest(1);
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {exec_req};

  exec_req->spec_state.draft_tokens.spec = {7, 8, 9};
  exec_req->spec_state.forwarding_tokens_draft_num = 0;
  exec_req->sampling_result_tokens = {42};

  // Step 1: Explicitly clear stale drafts (simulates HandlePostprocess logic).
  for (auto& req : reqs) {
    if (req->spec_state.forwarding_tokens_draft_num == 0) {
      req->spec_state.draft_tokens.clear();
    }
  }
  EXPECT_EQ(exec_req->spec_state.draft_tokens.size(), 0);

  // Step 2: DefaultFilterDraftTokens afterwards — sizes match, no mismatch triggered.
  size_t accepted = DefaultFilterDraftTokens(exec_req);
  EXPECT_EQ(accepted, 0);
  EXPECT_TRUE(exec_req->sampling_result_tokens.empty());
  EXPECT_EQ(exec_req->generated_tokens.size(), 1);
  EXPECT_EQ(exec_req->generated_tokens[0], 42);
}

// Regression test: DefaultFilterDraftTokens with mismatched sizes (stale drafts
// not cleared) triggers defensive mismatch check and returns hit=0.
// This validates that clearing stale drafts BEFORE verification is critical —
// a bug order (verify before clear) would produce this mismatch.
TEST_F(GenerationControllerTest, DefaultFilterDraftTokensMismatch) {
  auto exec_req = CreateMockExecutorInferRequest(1);

  exec_req->spec_state.draft_tokens.spec = {7, 8, 9};
  exec_req->spec_state.forwarding_tokens_draft_num = 0;
  exec_req->sampling_result_tokens = {42};

  // Call DefaultFilterDraftTokens WITHOUT first clearing stale drafts.
  // This simulates the bug order: verify before clear.
  // sampling_result_tokens.size()-1 (0) != draft_tokens.size() (3) → mismatch.
  size_t accepted = DefaultFilterDraftTokens(exec_req);
  EXPECT_EQ(accepted, 0);

  // Drafts are intact (no clear + no accept).
  EXPECT_EQ(exec_req->spec_state.draft_tokens.spec.size(), 3);
}

}  // namespace ksana_llm
