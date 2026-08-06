/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/speculative/common/draft_generator_interface.h"
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
  req->spec_state.accepted_tokens.clear();
  if (req->sampling_result_tokens.empty() ||
      (req->sampling_result_tokens.size() - kStepGenerateTokenNum != req->spec_state.draft_tokens.size() &&
       req->logits_custom_length == 0)) {
    KLLM_LOG_ERROR << fmt::format(
        "req {} sampling_result_tokens.size = {}, mtp_draft_tokens.size = {}, spec_draft_tokens.size = {}, "
        "ptp_draft_tokens.size = {}",
        req->req_id, req->sampling_result_tokens.size(), req->spec_state.draft_tokens.mtp.size(),
        req->spec_state.draft_tokens.spec.size(), req->spec_state.draft_tokens.ptp.size());
    return 0;
  }

  // Check which tokens are predicted correctly
  size_t draft_hit_num = 0;
  std::vector<int> draft_tokens = req->spec_state.draft_tokens.GetDraftTokens();

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
    if (req->structured_generator) {
      // Check structured_generator for new-style structured generation (xgrammar)
      if (req->structured_generator->IsTerminated()) {
        break;
      }
      const int new_token = req->sampling_result_tokens[i + 1];
      if (!req->structured_generator->AcceptToken(new_token)) {
        break;
      }
    }
  }

  const size_t total_draft_count = draft_tokens.size();

  req->spec_state.accepted_tokens.swap(draft_tokens);
  req->spec_state.accepted_tokens.resize(draft_hit_num);

  // Log acceptance rate.
  KLLM_LOG_DEBUG << "Draft verification: req_id=" << req->req_id << ", hit=" << draft_hit_num << "/"
                 << total_draft_count
                 << " (acceptance rate: " << (total_draft_count > 0 ? draft_hit_num * 100 / total_draft_count : 0)
                 << "%)";

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
