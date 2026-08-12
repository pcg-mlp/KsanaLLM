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

// =============================================================================
// build-ahead inflight 深度与 placeholder 单测
// -----------------------------------------------------------------------------
// 共享 helper：让 req 进入 decode 稳态（prefill 已 retire，inflight_count_=0），
// 之后可 SetupInflightDecodeTask + LaunchPlanningTask(use_placeholder=true) 模拟深度 2 push。
// =============================================================================
static std::shared_ptr<InferRequest> EnterDecodeSteadyState(InferRequestTest* fixture, int first_decoded_token) {
  auto req = fixture->CreateInferRequest({1, 2, 3});
  req->output_tokens = {1, 2, 3};
  req->ResetPrefillingTokens();
  // 一步 prefill 完成
  fixture->SetupInflightPrefillTask(req, /*prefill_token_num=*/3, /*prefill_start_offset=*/0);
  req->generated_tokens = {first_decoded_token};
  req->UpdateAfterInflightTaskFinished();
  req->ResetInflightTask();
  EXPECT_EQ(req->infer_stage, InferStage::kDecode);
  EXPECT_EQ(req->GetInflightCount(), 0u);
  return req;
}

// 首步 decode launch（无 inflight）不应写 placeholder，直接 append generated_tokens。
TEST_F(InferRequestTest, FastPath_FirstDecodeLaunchNoPlaceholder) {
  auto req = EnterDecodeSteadyState(this, /*first_decoded_token=*/10);
  ASSERT_EQ(req->GetInflightCount(), 0u);

  req->generated_tokens = {88};
  SetupInflightDecodeTask(req, /*generated_token_num=*/1, /*draft_token_num=*/0);
  ASSERT_EQ(req->GetInflightCount(), 1u);
  EXPECT_EQ(req->forwarding_tokens.back(), 88);
  EXPECT_NE(req->forwarding_tokens.back(), InferRequest::kFastPathPlaceholderTokenId);
}

// placeholder 回填后 generated_tokens 必须清空，避免下次 launch 重复 append。
TEST_F(InferRequestTest, FastPath_PlaceholderBackfillClearsGeneratedTokens) {
  auto req = EnterDecodeSteadyState(this, /*first_decoded_token=*/10);
  SetupInflightDecodeTask(req, 1, 0);

  req->generated_tokens.clear();
  ScheduleTaskWorkload wl;
  wl.generated_token_num = 1;
  wl.draft_token_num = 0;
  wl.sampling_token_num = 1;
  req->SetPlanningWorkload(wl);
  req->SetPlanningTask();
  req->LaunchPlanningTask(/*use_placeholder_for_last_token=*/true);

  req->generated_tokens = {66};
  req->UpdateAfterInflightTaskFinished();
  EXPECT_TRUE(req->generated_tokens.empty());
}

// ResetInflightTask 按 FIFO 弹出 oldest inflight（深度 2 时 [0] retire 后 [1] 滑到 [0]）。
TEST_F(InferRequestTest, FastPath_ResetInflightTaskFifoOrder) {
  auto req = EnterDecodeSteadyState(this, /*first_decoded_token=*/10);
  SetupInflightDecodeTask(req, 1, 0);

  req->generated_tokens.clear();
  ScheduleTaskWorkload wl;
  wl.generated_token_num = 1;
  wl.draft_token_num = 0;
  wl.sampling_token_num = 1;
  req->SetPlanningWorkload(wl);
  req->SetPlanningTask();
  req->LaunchPlanningTask(/*use_placeholder_for_last_token=*/true);
  ASSERT_EQ(req->GetInflightCount(), 2u);

  req->ResetInflightTask();
  EXPECT_EQ(req->GetInflightCount(), 1u);
  // 剩余 inflight 为 placeholder 步（step A 尚未 UpdateAfterInflightTaskFinished）
  EXPECT_EQ(req->forwarding_tokens.back(), InferRequest::kFastPathPlaceholderTokenId);
}

// HasInflightCapacity：深度 2 下精确放行 1 步，第 2 步（N+2 须等 N 返回）必须被拦截。
TEST_F(InferRequestTest, FastPath_HasInflightCapacityRespectsMaxDepth) {
  auto req = EnterDecodeSteadyState(this, /*first_decoded_token=*/10);
  // step k launched (slow path 风格, 不打 placeholder)
  SetupInflightDecodeTask(req, /*generated_token_num=*/1, /*draft_token_num=*/0);
  ASSERT_EQ(req->GetInflightCount(), 1u);

  // slow path 视角: 已无 capacity (max_depth=1)
  EXPECT_FALSE(req->HasInflightCapacity(1));
  // build-ahead 视角：还能再 push 1 个（max_depth=2）
  EXPECT_TRUE(req->HasInflightCapacity(2));

  // step k+1 以 placeholder launch（深度 2 第二次 launch）。
  // 真实场景：plan 阶段已 clear generated_tokens，step k 仍在 inflight、Tk 未回 Engine，故走 placeholder 路径。
  req->generated_tokens.clear();
  ScheduleTaskWorkload wl;
  wl.generated_token_num = 1;
  wl.draft_token_num = 0;
  wl.sampling_token_num = 1;
  req->SetPlanningWorkload(wl);
  req->SetPlanningTask();
  req->LaunchPlanningTask(/*use_placeholder_for_last_token=*/true);

  ASSERT_EQ(req->GetInflightCount(), 2u);
  // 深度 2 下 capacity 已用完，N+2 须等 N retire 才能 launch
  EXPECT_FALSE(req->HasInflightCapacity(2));
}

// LaunchPlanningTask(use_placeholder=true)：末尾写占位符并记录位置；
// UpdateAfterInflightTaskFinished 用真实 generated_token 回填，验证替换不变量。
TEST_F(InferRequestTest, FastPath_LaunchPlaceholderThenReplaceOnRetire) {
  auto req = EnterDecodeSteadyState(this, /*first_decoded_token=*/10);
  // step k launched: forwarding_tokens 末尾是真实 token 10 (来自 prefill retire)
  SetupInflightDecodeTask(req, /*generated_token_num=*/1, /*draft_token_num=*/0);
  ASSERT_EQ(req->GetInflightCount(), 1u);
  ASSERT_EQ(req->forwarding_tokens.back(), 10);

  // step k+1 launched with PLACEHOLDER (T_k 未回).
  // 模拟 ScheduleProcessor 已经 clear generated_tokens (此时 step k 真实 token 还没回 Engine).
  req->generated_tokens.clear();
  ScheduleTaskWorkload wl;
  wl.generated_token_num = 1;
  wl.draft_token_num = 0;
  wl.sampling_token_num = 1;
  req->SetPlanningWorkload(wl);
  req->SetPlanningTask();
  req->LaunchPlanningTask(/*use_placeholder_for_last_token=*/true);

  // forwarding_tokens 末尾应是 kFastPathPlaceholderTokenId
  ASSERT_EQ(req->forwarding_tokens.back(), InferRequest::kFastPathPlaceholderTokenId);
  const size_t placeholder_pos = req->forwarding_tokens.size() - 1;

  // step k IPC 回来: 真实 Tk = 55
  req->generated_tokens = {55};
  req->UpdateAfterInflightTaskFinished();

  // placeholder 已经被替换为 55
  EXPECT_EQ(req->forwarding_tokens[placeholder_pos], 55);

  // retire oldest, inflight 滑动: 2 → 1
  req->ResetInflightTask();
  EXPECT_EQ(req->GetInflightCount(), 1u);
}

// logits_custom_length > 1（slice_pos 覆盖多个 token）时，LaunchPlanningTask 必须把
// inflight workload 与 InferRequest::sampling_token_num 抬到 logits_custom_length，
// 否则下游 ModelInput 只按 1 分配 logits_idx_list，AssembleTokensHidden 输出 shape[0]=1。
TEST_F(InferRequestTest, LaunchPlanningTaskPreservesLogitsCustomLength_Prefill) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5});
  req->output_tokens = {1, 2, 3, 4, 5};
  req->logits_custom_length = 3;  // 例如 slice_pos = {{0,2}}
  req->ResetPrefillingTokens();
  ASSERT_EQ(req->sampling_token_num, 3u);

  SetupInflightPrefillTask(req, /*prefill_token_num=*/5, /*prefill_start_offset=*/0);
  ASSERT_FALSE(req->IsSplitPrefillStep());

  EXPECT_EQ(req->sampling_token_num, 3u);
  EXPECT_EQ(req->GetInflightTask().workload.sampling_token_num, 3u);
}

// RetireChunkedReqs：清空 oldest inflight，remaining prefill 迁到 planning_workload_，
// remaining_workload_.generated/draft 归零。同时验证 forwarding_tokens 包含正确的
// chunk 前缀 token、prefill_start_offset 正确推进。
TEST_F(InferRequestTest, RetireChunkedReqs_ResetsInflightAndUpdatesWorkload) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5, 6});
  req->output_tokens = {1, 2, 3, 4, 5, 6};
  req->ResetPrefillingTokens();

  // First chunk: 2 out of 6 → split step. remaining = 4.
  SetupInflightPrefillTask(req, /*prefill_token_num=*/2, /*prefill_start_offset=*/0);
  ASSERT_EQ(req->GetInflightCount(), 1u);
  ASSERT_TRUE(req->IsSplitPrefillStep());

  // 第一块 forwarding_tokens 应包含前 2 个 token。
  ASSERT_EQ(req->forwarding_tokens, std::vector<int>({1, 2}));
  // remaining_workload_ 的 offset 已由 SetPlanningTask 推进。
  ASSERT_EQ(req->GetRemainingWorkload().prefill_start_offset, 2u);

  const size_t fwd_before = req->forwarding_tokens.size();
  req->RetireChunkedReqs();

  EXPECT_EQ(req->GetInflightCount(), 0u);
  // forwarding_tokens 不变；kv_cached 推进到当前 forwarding_tokens 尾部
  EXPECT_EQ(req->forwarding_tokens.size(), fwd_before);
  EXPECT_EQ(req->forwarding_tokens, std::vector<int>({1, 2}));
  EXPECT_EQ(req->kv_cached_token_num, fwd_before);
  // planning_workload_ 承载余下 4 个 prefill token；prefill_start_offset 已 Reset 归零
  EXPECT_EQ(req->GetPlanningWorkload().prefill_token_num, 4u);
  EXPECT_EQ(req->GetPlanningWorkload().prefill_start_offset, 0u);
  // remaining_workload_ offset 保持推进（不受 Retire 影响）
  EXPECT_EQ(req->GetRemainingWorkload().prefill_start_offset, 2u);
  EXPECT_EQ(req->GetRemainingWorkload().generated_token_num, 0u);
  EXPECT_EQ(req->GetRemainingWorkload().draft_token_num, 0u);
}

// RetireChunkedReqs 之后：下一次 UpdateAfterInflightTaskFinished 必须返回 false，
// 清掉本次 sample 回流的 generated/accepted/draft，且不推 output_tokens。
TEST_F(InferRequestTest, RetireChunkedReqs_NextUpdateSkipsSampleFlush) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5, 6});
  req->output_tokens = {1, 2, 3, 4, 5, 6};
  req->ResetPrefillingTokens();

  SetupInflightPrefillTask(req, /*prefill_token_num=*/2, /*prefill_start_offset=*/0);
  req->RetireChunkedReqs();

  // 模拟：sample 回流带来 stale 结果
  req->generated_tokens = {77};
  req->accepted_tokens = {88};
  req->draft_tokens.mtp = {99};
  const size_t output_before = req->output_tokens.size();

  bool consumed = req->UpdateAfterInflightTaskFinished();

  EXPECT_FALSE(consumed);
  EXPECT_TRUE(req->generated_tokens.empty());
  EXPECT_TRUE(req->accepted_tokens.empty());
  EXPECT_EQ(req->draft_tokens.size(), 0u);
  EXPECT_EQ(req->output_tokens.size(), output_before);
}

// RetireChunkedReqs 之后重新 Launch 下一个 chunk：验证 forwarding_tokens 连续拼接、
// kv_cached 正确推进，形成完整的 chunked prefill 端到端流程。
TEST_F(InferRequestTest, RetireChunkedReqs_NextChunkPreservesTokenContinuity) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5, 6});
  req->output_tokens = {1, 2, 3, 4, 5, 6};
  req->ResetPrefillingTokens();

  // ----- 第一块：prefill 2 token（offset 0）-----
  SetupInflightPrefillTask(req, /*prefill_token_num=*/2, /*prefill_start_offset=*/0);
  ASSERT_EQ(req->forwarding_tokens, std::vector<int>({1, 2}));
  ASSERT_EQ(req->kv_cached_token_num, 0u);  // Launch 时 kv_cached 取 forwarding_tokens 旧 size

  req->RetireChunkedReqs();
  ASSERT_EQ(req->GetInflightCount(), 0u);
  // Retire 把 kv_cached 推进到 forwarding_tokens 尾部
  ASSERT_EQ(req->kv_cached_token_num, 2u);
  // remaining_workload_ 仍记录已推进的 offset
  ASSERT_EQ(req->GetRemainingWorkload().prefill_start_offset, 2u);
  ASSERT_EQ(req->GetRemainingWorkload().prefill_token_num, 4u);

  // ----- 第二块：prefill 2 token（offset 2）-----
  SetupInflightPrefillTask(req, /*prefill_token_num=*/2, /*prefill_start_offset=*/2);
  ASSERT_EQ(req->GetInflightCount(), 1u);
  // forwarding_tokens 拼接了第一块 + 第二块，完整连续
  EXPECT_EQ(req->forwarding_tokens, std::vector<int>({1, 2, 3, 4}));
  // kv_cached 在 Launch 时再次被设为 forwarding_tokens 旧 size（=2），
  // 表示前 2 个 token 已落 KV cache，后 2 个在飞
  EXPECT_EQ(req->kv_cached_token_num, 2u);
  // remaining offset 已推进到 4（第二块结束位置）
  EXPECT_EQ(req->GetRemainingWorkload().prefill_start_offset, 4u);
  EXPECT_EQ(req->GetRemainingWorkload().prefill_token_num, 2u);

  // ----- 第二块 Retire -----
  req->RetireChunkedReqs();
  EXPECT_EQ(req->GetInflightCount(), 0u);
  EXPECT_EQ(req->forwarding_tokens, std::vector<int>({1, 2, 3, 4}));
  EXPECT_EQ(req->kv_cached_token_num, 4u);
  // 剩余 2 个 prefill 迁入 planning_workload_
  EXPECT_EQ(req->GetPlanningWorkload().prefill_token_num, 2u);
}

// 正常 decode step 返回 true（区别于 RetireChunkedReqs 分支）。
TEST_F(InferRequestTest, UpdateAfterInflightTaskFinished_ReturnsTrueOnNormalDecode) {
  auto req = CreateInferRequest({1, 2, 3});
  req->output_tokens = {1, 2, 3};
  req->ResetPrefillingTokens();

  SetupInflightPrefillTask(req, /*prefill_token_num=*/3, /*prefill_start_offset=*/0);
  req->generated_tokens = {10};
  EXPECT_TRUE(req->UpdateAfterInflightTaskFinished());
}

// build-ahead 下 step N 产出 EOS 后，N+1 占位步必须丢弃，不写 output_tokens、不回填 placeholder。
TEST_F(InferRequestTest, FastPath_EosDropsExtraStep) {
  const int eos = 999;
  auto req = CreateInferRequest({1, 2, 3}, /*stop_token_ids=*/{eos});
  req->output_tokens = {1, 2, 3};
  req->ResetPrefillingTokens();
  SetupInflightPrefillTask(req, /*prefill_token_num=*/3, /*prefill_start_offset=*/0);
  req->generated_tokens = {10};
  req->UpdateAfterInflightTaskFinished();
  req->ResetInflightTask();

  // step k launched (decode, 真实 token=10 在末位)
  SetupInflightDecodeTask(req, 1, 0);
  ASSERT_EQ(req->GetInflightCount(), 1u);

  // step k+1 launched with PLACEHOLDER. 模拟 ScheduleProcessor 已 clear generated_tokens.
  req->generated_tokens.clear();
  ScheduleTaskWorkload wl;
  wl.generated_token_num = 1;
  wl.draft_token_num = 0;
  wl.sampling_token_num = 1;
  req->SetPlanningWorkload(wl);
  req->SetPlanningTask();
  req->LaunchPlanningTask(/*use_placeholder_for_last_token=*/true);
  ASSERT_EQ(req->GetInflightCount(), 2u);

  // step k retire: 真实 Tk = EOS
  req->generated_tokens = {eos};
  size_t output_before = req->output_tokens.size();
  req->UpdateAfterInflightTaskFinished();
  req->ResetInflightTask();
  EXPECT_TRUE(req->IsEosGenerated());
  EXPECT_EQ(req->output_tokens.back(), eos);
  EXPECT_EQ(req->output_tokens.size(), output_before + 1);

  // step k+1 retire: 应丢弃 (即使 generated_tokens 有值)
  req->generated_tokens = {12345};
  size_t output_before2 = req->output_tokens.size();
  size_t fwd_before2 = req->forwarding_tokens.size();
  req->UpdateAfterInflightTaskFinished();
  req->ResetInflightTask();
  EXPECT_EQ(req->GetInflightCount(), 0u);
  // EOS 多算一步: output_tokens 不变, forwarding_tokens 不再被推进
  EXPECT_EQ(req->output_tokens.size(), output_before2);
  EXPECT_EQ(req->forwarding_tokens.size(), fwd_before2);
}

}  // namespace ksana_llm
