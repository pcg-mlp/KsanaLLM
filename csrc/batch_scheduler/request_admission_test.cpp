/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/request_admission.h"

#include <memory>
#include <unordered_map>

#include "test.h"
#include "utils/request.h"

namespace ksana_llm {
namespace {

struct AdmissionTestRequest {
  std::shared_ptr<Request> request;
  std::shared_ptr<InferRequest> infer_request;
};

AdmissionTestRequest MakeAdmissionTestRequest(size_t input_len) {
  auto input = std::make_shared<KsanaPythonInput>();
  input->input_tokens.resize(input_len);
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, req_ctx);
  auto infer_request = std::make_shared<InferRequest>(request, 0);
  return {request, infer_request};
}

TEST(RequestAdmissionTest, AcceptsBoundaryValues) {
  BatchSchedulerConfig config;
  config.max_token_len = 16;
  config.max_batch_size = 8;
  auto req = MakeAdmissionTestRequest(config.max_token_len);
  req.infer_request->logits_custom_length = config.max_batch_size;

  EXPECT_TRUE(ValidateRequestAdmission(req.infer_request, config).OK());
}

TEST(RequestAdmissionTest, RejectsInputLengthAboveLimit) {
  BatchSchedulerConfig config;
  config.max_token_len = 16;
  config.max_batch_size = 8;
  auto req = MakeAdmissionTestRequest(config.max_token_len + 1);

  EXPECT_EQ(ValidateRequestAdmission(req.infer_request, config).GetCode(), RET_INPUT_LENGTH_EXCEEDED);
}

TEST(RequestAdmissionTest, RejectsLogitsLengthAboveEffectiveLimit) {
  BatchSchedulerConfig config;
  config.max_token_len = 16;
  config.max_batch_size = 8;
  auto req = MakeAdmissionTestRequest(/*input_len=*/4);
  req.infer_request->logits_custom_length = 5;

  EXPECT_EQ(ValidateRequestAdmission(req.infer_request, config).GetCode(), RET_INPUT_LENGTH_EXCEEDED);
}

}  // namespace
}  // namespace ksana_llm
