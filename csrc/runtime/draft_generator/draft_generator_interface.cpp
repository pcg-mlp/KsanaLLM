/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/draft_generator/draft_generator_interface.h"
#include "fmt/format.h"
#include "runtime/executor_infer_request.h"
#include "utils/logger.h"

namespace ksana_llm {

// Default implementation: delegates to the standalone function
size_t DraftGeneratorInterface::FilterDraftTokens(std::shared_ptr<ExecutorInferRequest> req) {
  return DefaultFilterDraftTokens(req);
}

// Default draft token filtering logic (extracted from model_runner.cpp)
size_t DefaultFilterDraftTokens(std::shared_ptr<ExecutorInferRequest> req) {
  req->accepted_tokens.clear();
  if (req->sampling_result_tokens.size() - kStepGenerateTokenNum != req->draft_tokens.size() &&
      req->logits_custom_length == 0) {
    KLLM_LOG_ERROR << fmt::format(
        "req {} sampling_result_tokens.size = {}, mtp_draft_tokens.size = {}, spec_draft_tokens.size = {}, "
        "ptp_draft_tokens.size = {}",
        req->req_id, req->sampling_result_tokens.size(), req->draft_tokens.mtp.size(), req->draft_tokens.spec.size(),
        req->draft_tokens.ptp.size());
    return 0;
  }

  // Check which tokens are predicted correctly
  size_t draft_hit_num = 0;
  std::vector<int> draft_tokens = req->draft_tokens.GetDraftTokens();

  for (size_t i = 0; i < draft_tokens.size(); ++i, ++draft_hit_num) {
    if (req->sampling_result_tokens[i] != draft_tokens[i]) {
      break;
    }

    // stop if stop token
    if (std::find(req->sampling_config.stop_token_ids.begin(), req->sampling_config.stop_token_ids.end(),
                  draft_tokens[i]) != req->sampling_config.stop_token_ids.end()) {
      break;
    }

    // Check grammar matcher constraint for the next sampling result token
    if (req->grammar_matcher) {
      // Check if the next sampling result token (newly generated) is acceptable
      if (!req->grammar_matcher->AcceptToken(req->sampling_result_tokens[i + 1])) {
        // Next token is rejected by grammar matcher, stop here
        break;
      }
      // Rollback the tentative acceptance since we're just checking
      req->grammar_matcher->Rollback(1);
    }
  }

  req->accepted_tokens.swap(draft_tokens);
  req->accepted_tokens.resize(draft_hit_num);

  // Delete logprobs of tokens that were not accepted
  size_t unaccepted_num = req->sampling_result_tokens.size() - kStepGenerateTokenNum - draft_hit_num;
  for (size_t i = 0; i < unaccepted_num; ++i) {
    if (!req->logprobs.empty()) {
      req->logprobs.pop_back();
    }
  }

  // Replace generated_token with draft token's next token
  req->generated_tokens.clear();
  req->generated_tokens.push_back(req->sampling_result_tokens[draft_hit_num]);
  req->sampling_result_tokens.clear();

  return draft_hit_num;
}

}  // namespace ksana_llm
