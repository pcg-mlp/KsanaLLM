/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/

#include <gtest/gtest.h>

#include "ksana_llm/utils/request.h"

namespace ksana_llm {

class RequestTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// Test for `VerifyArgs()` of `SamplingConfig`.
TEST_F(RequestTest, SamplingConfig) {
  SamplingConfig sampling_config;
  EXPECT_TRUE(sampling_config.VerifyArgs().OK());

  sampling_config.topp = 0.f;
  EXPECT_TRUE(sampling_config.VerifyArgs().OK());
  EXPECT_EQ(sampling_config.topp, 1.f);

  sampling_config.temperature = 0.f;
  EXPECT_TRUE(sampling_config.VerifyArgs().OK());
  EXPECT_EQ(sampling_config.temperature, 1.f);

  sampling_config.topk = 0;
  EXPECT_EQ(sampling_config.VerifyArgs().GetCode(), RET_INVALID_ARGUMENT);

  sampling_config.topk = 1234;
  EXPECT_EQ(sampling_config.VerifyArgs().GetCode(), RET_INVALID_ARGUMENT);

  sampling_config.topk = 1024;
  sampling_config.no_repeat_ngram_size = sampling_config.encoder_no_repeat_ngram_size = 1;
  EXPECT_EQ(sampling_config.VerifyArgs().GetCode(), RET_INVALID_ARGUMENT);
}

}  // namespace ksana_llm