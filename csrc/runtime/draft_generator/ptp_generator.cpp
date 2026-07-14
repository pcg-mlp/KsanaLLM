/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/draft_generator/ptp_generator.h"
#include "runtime/executor_infer_request.h"

namespace ksana_llm {

PtpGenerator::PtpGenerator(size_t ptp_step_num, uint32_t ptp_token_id)
    : ptp_step_num_(ptp_step_num), ptp_token_id_(ptp_token_id) {}

void PtpGenerator::GenerateDraft(std::shared_ptr<ExecutorInferRequest> req) {
  for (size_t i = 0; i < ptp_step_num_; ++i) {
    req->draft_tokens.ptp.emplace_back(ptp_token_id_);
  }
}

size_t PtpGenerator::FilterDraftTokens(std::shared_ptr<ExecutorInferRequest> req) {
  // In PTP mode we keep none of the draft tokens.
  req->accepted_tokens.clear();

  // Transfer all sampling_result_tokens to generated_tokens
  // sampling_result_tokens contains: [real_token, ptp_draft_token_1, ..., ptp_draft_token_n]
  // We keep all of them as generated_tokens for planning_workload_ calculation
  req->generated_tokens.clear();
  req->generated_tokens.swap(req->sampling_result_tokens);
  req->draft_tokens.clear();
  return 0;
}

}  // namespace ksana_llm