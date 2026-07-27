/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/request_admission.h"

#include <algorithm>

#include "utils/ret_code.h"

namespace ksana_llm {

Status ValidateRequestAdmission(const std::shared_ptr<InferRequest>& req, const BatchSchedulerConfig& config) {
  const size_t input_len = req->input_tokens.size();
  const size_t max_logits_custom_length = std::min(input_len, config.max_batch_size);
  if (input_len > config.max_token_len || req->logits_custom_length > max_logits_custom_length) {
    return Status(RET_INPUT_LENGTH_EXCEEDED, "input length or logits_custom_length exceeds the limit.");
  }
  return Status();
}

}  // namespace ksana_llm
