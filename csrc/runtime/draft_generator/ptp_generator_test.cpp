/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/draft_generator/ptp_generator.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "runtime/executor_infer_request.h"

namespace ksana_llm {

// Test for PtpGenerator::FilterDraftTokens
class PtpGeneratorTest : public testing::Test {
 protected:
  void SetUp() override {
    // 创建 PtpGenerator，ptp_step_num=3, ptp_token_id=999
    generator_ = std::make_shared<PtpGenerator>(3, 999);
  }

  std::shared_ptr<ExecutorInferRequest> CreateMockInferRequest(const std::vector<int>& draft_tokens_ptp,
                                                               const std::vector<int>& sampling_result_tokens) {
    auto executor_req = std::make_shared<ExecutorInferRequest>();

    executor_req->req_id = 1;
    executor_req->input_tokens = {1, 2, 3};
    executor_req->output_tokens = {};
    executor_req->finished = false;
    executor_req->aborted = false;

    executor_req->draft_tokens.ptp = draft_tokens_ptp;
    executor_req->sampling_result_tokens = sampling_result_tokens;

    return executor_req;
  }

  std::shared_ptr<PtpGenerator> generator_;
};

TEST_F(PtpGeneratorTest, GenerateDraft) {
  // 测试 PtpGenerator::GenerateDraft
  auto req = std::make_shared<ExecutorInferRequest>();
  req->req_id = 1;

  generator_->GenerateDraft(req);

  // 验证生成了 3 个 ptp draft tokens，每个都是 999
  EXPECT_EQ(req->draft_tokens.ptp.size(), 3);
  EXPECT_EQ(req->draft_tokens.ptp[0], 999);
  EXPECT_EQ(req->draft_tokens.ptp[1], 999);
  EXPECT_EQ(req->draft_tokens.ptp[2], 999);
}

TEST_F(PtpGeneratorTest, FilterDraftTokens) {
  // 测试 PtpGenerator::FilterDraftTokens
  std::vector<int> draft_ptp = {999, 999, 999};
  std::vector<int> sampling_result = {10, 20, 30, 40};

  auto req = CreateMockInferRequest(draft_ptp, sampling_result);

  // 调用 FilterDraftTokens
  size_t accepted_num = generator_->FilterDraftTokens(req);

  // 验证返回值为 0（PTP 模式不接受任何 draft tokens）
  EXPECT_EQ(accepted_num, 0);
  // 验证 accepted_tokens 为空
  EXPECT_TRUE(req->accepted_tokens.empty());
  // 验证 generated_tokens 包含所有 sampling_result_tokens
  EXPECT_EQ(req->generated_tokens.size(), 4);
  EXPECT_EQ(req->generated_tokens[0], 10);
  EXPECT_EQ(req->generated_tokens[1], 20);
  EXPECT_EQ(req->generated_tokens[2], 30);
  EXPECT_EQ(req->generated_tokens[3], 40);
  // 验证 sampling_result_tokens 被清空
  EXPECT_TRUE(req->sampling_result_tokens.empty());
  // 验证 draft_tokens 被清空
  EXPECT_EQ(req->draft_tokens.size(), 0);
}

}  // namespace ksana_llm