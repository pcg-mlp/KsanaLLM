/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/draft_generator/draft_generator_controller.h"
#include <gtest/gtest.h>
#include "runtime/draft_generator/ptp_generator.h"
#include "runtime/executor_infer_request.h"

namespace ksana_llm {

class DraftGeneratorControllerTest : public testing::Test {
 protected:
  void SetUp() override { draft_generator_controller_ = std::make_shared<DraftGeneratorController>(); }

 protected:
  std::shared_ptr<DraftGeneratorController> draft_generator_controller_;
};

TEST_F(DraftGeneratorControllerTest, PTPGeneratorTest) {
  // 验证 Append 功能
  Status status = draft_generator_controller_->AppendDraftGenerator(std::make_shared<PtpGenerator>(3, 1234));
  EXPECT_TRUE(status.OK());
  status = draft_generator_controller_->AppendDraftGenerator(std::make_shared<PtpGenerator>(2, 2345));
  EXPECT_TRUE(status.OK());

  // 创建 ExecutorInferRequest 用于测试
  auto executor_infer_request = std::make_shared<ExecutorInferRequest>();
  executor_infer_request->req_id = 1;
  executor_infer_request->draft_tokens.clear();

  // 验证顺序 Generator 推理
  status = draft_generator_controller_->GenerateDraft(executor_infer_request);
  EXPECT_TRUE(status.OK());

  std::vector<int> target_draft_tokens = {1234, 1234, 1234, 2345, 2345};
  std::vector<int> draft_tokens = executor_infer_request->draft_tokens.GetDraftTokens();
  EXPECT_EQ(draft_tokens.size(), target_draft_tokens.size());
  for (size_t i = 0; i < target_draft_tokens.size(); ++i) {
    EXPECT_EQ(draft_tokens[i], target_draft_tokens[i]);
  }
}

}  // namespace ksana_llm