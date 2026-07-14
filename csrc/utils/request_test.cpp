/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/

#include <gtest/gtest.h>

#include "endpoints/streaming/streaming_iterator.h"
#include "utils/request.h"
#include "utils/status.h"

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

TEST_F(RequestTest, RequestStatus) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx =
      std::make_shared<std::unordered_map<std::string, std::string>>();

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);
  request->finish_status = Status(RET_SUCCESS, "success");
  auto ksana_python_output = KsanaPythonOutput(request);
  EXPECT_EQ(ksana_python_output.finish_status.GetCode(), RetCode::RET_SUCCESS);
}

// Test for prefix_cache_hit_tokens in KsanaPythonOutput
TEST_F(RequestTest, PrefixCacheHitTokens) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx =
      std::make_shared<std::unordered_map<std::string, std::string>>();

  // Test case 1: No prefix_cache_hit_tokens in req_ctx
  {
    std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);
    auto ksana_python_output = KsanaPythonOutput(request);
    EXPECT_EQ(ksana_python_output.prefix_cache_hit_tokens, 0);
  }

  // Test case 2: With prefix_cache_hit_tokens in req_ctx
  {
    (*req_ctx)["prefix_cache_hit_tokens"] = "42";
    std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);
    auto ksana_python_output = KsanaPythonOutput(request);
    EXPECT_EQ(ksana_python_output.prefix_cache_hit_tokens, 42);
  }

  // Test case 3: With different prefix_cache_hit_tokens value
  {
    (*req_ctx)["prefix_cache_hit_tokens"] = "128";
    std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);
    auto ksana_python_output = KsanaPythonOutput(request);
    EXPECT_EQ(ksana_python_output.prefix_cache_hit_tokens, 128);
  }

  // Test case 4: req_ctx is nullptr
  {
    std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, nullptr);
    auto ksana_python_output = KsanaPythonOutput(request);
    EXPECT_EQ(ksana_python_output.prefix_cache_hit_tokens, 0);
  }
}

// Test for StreamingIterator with prefix_cache_hit_tokens
TEST_F(RequestTest, StreamingIteratorPrefixCacheHitTokens) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3, 4, 5};
  ksana_python_input->sampling_config.logprobs_num = 0;

  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx =
      std::make_shared<std::unordered_map<std::string, std::string>>();
  (*req_ctx)["prefix_cache_hit_tokens"] = "50";

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);
  // Initialize required members for StreamingIterator
  request->finisheds = {true};
  request->abort_waiter = std::make_shared<Waiter>(0);
  request->output_group.push_back(std::make_tuple(
      std::vector<int>{1, 2, 3, 4, 5, 6, 7},
      std::vector<std::vector<std::pair<int, float>>>(),
      0.0f));

  StreamingIterator iterator(request, ksana_python_input);

  KsanaPythonOutput output;
  bool has_output = iterator.AddOutput(output);

  EXPECT_TRUE(has_output);
  EXPECT_EQ(output.prefix_cache_hit_tokens, 50);
  EXPECT_EQ(output.input_tokens.size(), 5);
}

}  // namespace ksana_llm
