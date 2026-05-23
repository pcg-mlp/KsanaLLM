/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/full_cache_hit_handler.h"

#include <gtest/gtest.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "runtime/infer_request.h"
#include "utils/request.h"

namespace ksana_llm {

class FullCacheHitHandlerTest : public testing::Test {
 protected:
  void SetUp() override { InitLoguru(); }

  std::shared_ptr<InferRequest> CreateInferRequest(const std::vector<int>& input_tokens) {
    auto python_input = std::make_shared<KsanaPythonInput>();
    python_input->input_tokens = input_tokens;

    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    request_ = std::make_shared<Request>(python_input, req_ctx);
    python_input_ = python_input;

    return std::make_shared<InferRequest>(request_, 0);
  }

  std::shared_ptr<KsanaPythonInput> python_input_;
  std::shared_ptr<Request> request_;
};

// Non-SWA model: prefill_token_num = kStepGenerateTokenNum, stage switches to kDecode.
TEST_F(FullCacheHitHandlerTest, NonSwaFullHit) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5});
  req->output_tokens = {1, 2, 3, 4, 5};
  req->block_token_num = 16;
  // swa_kv_cache_blocks empty => non-SWA

  ScheduleTaskWorkload workload;
  workload.prefill_token_num = 5;
  req->SetPlanningWorkload(workload);
  req->SetRemainingWorkload(workload);

  HandleFullPrefixCacheHit(req, workload, /*seq_token_num=*/5);

  EXPECT_EQ(workload.prefill_start_offset, 5 - kStepGenerateTokenNum);
  EXPECT_EQ(workload.prefill_token_num, kStepGenerateTokenNum);
  EXPECT_EQ(req->infer_stage, InferStage::kDecode);
}

// SWA model: prefill_start_offset floor-aligned to block_token_num, stage stays unchanged.
// seq_token_num=10, block_token_num=4
// prefill_start_offset = (10-1)/4*4 = 8, prefill_token_num = 10 - 8 = 2
TEST_F(FullCacheHitHandlerTest, SwaFullHit) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  req->output_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  req->block_token_num = 4;
  req->swa_kv_cache_blocks = {0, 1, 2};  // non-empty => SWA enabled

  ScheduleTaskWorkload workload;
  workload.prefill_token_num = 10;
  req->SetPlanningWorkload(workload);
  req->SetRemainingWorkload(workload);

  const auto stage_before = req->infer_stage;
  HandleFullPrefixCacheHit(req, workload, /*seq_token_num=*/10);

  EXPECT_EQ(workload.prefill_start_offset, 8);
  EXPECT_EQ(workload.prefill_token_num, 2);
  EXPECT_EQ(req->infer_stage, stage_before);  // stage not changed for SWA
}

// SWA model: seq_token_num is exactly aligned to block_token_num.
// seq_token_num=8, block_token_num=4
// prefill_start_offset = (8-1)/4*4 = 4, prefill_token_num = 8 - 4 = 4
TEST_F(FullCacheHitHandlerTest, SwaFullHitAligned) {
  auto req = CreateInferRequest({1, 2, 3, 4, 5, 6, 7, 8});
  req->output_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
  req->block_token_num = 4;
  req->swa_kv_cache_blocks = {0, 1};  // non-empty => SWA enabled

  ScheduleTaskWorkload workload;
  workload.prefill_token_num = 8;
  req->SetPlanningWorkload(workload);
  req->SetRemainingWorkload(workload);

  const auto stage_before = req->infer_stage;
  HandleFullPrefixCacheHit(req, workload, /*seq_token_num=*/8);

  EXPECT_EQ(workload.prefill_start_offset, 4);
  EXPECT_EQ(workload.prefill_token_num, 4);
  EXPECT_EQ(req->infer_stage, stage_before);
}

// SWA model: seq_token_num < block_token_num, prefill the entire sequence.
TEST_F(FullCacheHitHandlerTest, SwaShortSequence) {
  auto req = CreateInferRequest({1, 2, 3});
  req->output_tokens = {1, 2, 3};
  req->block_token_num = 256;
  req->swa_kv_cache_blocks = {0};  // SWA enabled

  ScheduleTaskWorkload workload;
  workload.prefill_token_num = 3;
  req->SetPlanningWorkload(workload);
  req->SetRemainingWorkload(workload);

  HandleFullPrefixCacheHit(req, workload, /*seq_token_num=*/3);

  // seq_token_num <= block_token_num => prefill_start_offset = 0
  EXPECT_EQ(workload.prefill_start_offset, 0);
  EXPECT_EQ(workload.prefill_token_num, 3);
}

}  // namespace ksana_llm
