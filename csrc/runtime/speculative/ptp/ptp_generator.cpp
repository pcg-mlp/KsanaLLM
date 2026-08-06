/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/speculative/ptp/ptp_generator.h"
#include "runtime/executor_infer_request.h"

namespace ksana_llm {

PtpGenerator::PtpGenerator(size_t ptp_step_num, uint32_t ptp_token_id)
    : ptp_step_num_(ptp_step_num), ptp_token_id_(ptp_token_id) {}

SpecSchedulingInfo PtpGenerator::GetSchedulingInfo() const {
  SpecSchedulingInfo info;
  // PTP in steady state: ptp_step_num verify tokens + ptp_step_num placeholder tokens + 1 generated token
  // The extra_token_capacity accounts for the additional tokens beyond the single generated token
  info.extra_token_capacity = 2 * ptp_step_num_;
  info.workload_multiplier = 2 * ptp_step_num_;
  return info;
}

void PtpGenerator::GenerateDraft(std::shared_ptr<ExecutorInferRequest> req) {
  int per_req = req->sampling_config.ptp_token_num;
  size_t num =
      (per_req >= 0 && static_cast<size_t>(per_req) <= ptp_step_num_) ? static_cast<size_t>(per_req) : ptp_step_num_;
  for (size_t i = 0; i < num; ++i) {
    req->spec_state.draft_tokens.ptp.emplace_back(ptp_token_id_);
  }
  KLLM_LOG_DEBUG << "PTP GenerateDraft: req_id=" << req->req_id << ", ptp_tokens=" << num
                 << ", ptp_token_id=" << ptp_token_id_;
}

size_t PtpGenerator::FilterDraftTokens(std::shared_ptr<ExecutorInferRequest> req) {
  KLLM_LOG_DEBUG << "PTP FilterDraftTokens entry: req_id=" << req->req_id
                 << ", sampling_result_tokens.size=" << req->sampling_result_tokens.size()
                 << ", ptp.size=" << req->spec_state.draft_tokens.ptp.size()
                 << ", mtp.size=" << req->spec_state.draft_tokens.mtp.size()
                 << ", spec.size=" << req->spec_state.draft_tokens.spec.size();
  // Mixed mode (has MTP/Spec draft tokens): PTP should not handle them.
  // This should not happen after the mutual exclusion check in schedule_config_parser,
  // but as a safeguard, fall back to default filtering which handles all draft token types.
  if (!req->spec_state.draft_tokens.mtp.empty() || !req->spec_state.draft_tokens.spec.empty()) {
    return DefaultFilterDraftTokens(req);
  }

  req->spec_state.accepted_tokens.clear();

  // Precondition: sampling_result_tokens.size() - kStepGenerateTokenNum == draft_tokens.ptp.size()
  // Handle cold start: no draft tokens to verify yet.
  if (req->sampling_result_tokens.size() <= kStepGenerateTokenNum) {
    req->generated_tokens.clear();
    req->generated_tokens.swap(req->sampling_result_tokens);
    req->spec_state.draft_tokens.clear();
    return 0;
  }
  if (req->sampling_result_tokens.size() - kStepGenerateTokenNum != req->spec_state.draft_tokens.ptp.size()) {
    KLLM_LOG_ERROR << fmt::format("PTP req {} sampling_result_tokens.size = {}, ptp_draft_tokens.size = {}",
                                  req->req_id, req->sampling_result_tokens.size(),
                                  req->spec_state.draft_tokens.ptp.size());
    req->generated_tokens.clear();
    req->generated_tokens.swap(req->sampling_result_tokens);
    req->spec_state.draft_tokens.clear();
    return 0;
  }

  // Find the boundary between verify tokens and placeholder tokens
  size_t verify_count = 0;
  for (const auto& token : req->spec_state.draft_tokens.ptp) {
    if (static_cast<uint32_t>(token) == ptp_token_id_) break;
    verify_count++;
  }
  size_t ptp_placeholder_count = req->spec_state.draft_tokens.ptp.size() - verify_count;

  // ---- Pure guess round (verify_count == 0, all placeholders) ----
  if (verify_count == 0) {
    req->generated_tokens.clear();
    req->generated_tokens.push_back(req->sampling_result_tokens[0]);

    req->spec_state.draft_tokens.ptp.clear();
    for (size_t i = kStepGenerateTokenNum; i < req->sampling_result_tokens.size(); ++i) {
      req->spec_state.draft_tokens.ptp.push_back(req->sampling_result_tokens[i]);
    }

    // Pop logprobs for placeholder positions
    for (size_t i = 0; i < ptp_placeholder_count; ++i) {
      if (!req->logprobs.empty()) req->logprobs.pop_back();
    }
    req->sampling_result_tokens.clear();
    KLLM_LOG_DEBUG << "PTP FilterDraftTokens: req_id=" << req->req_id
                   << ", pure_guess_round, ptp_tokens=" << req->spec_state.draft_tokens.ptp.size();
    return 0;
  }

  // ---- Verification round ----
  // PTP draft tokens layout: [verify_token_0, ..., verify_token_(k-1), placeholder, ..., placeholder]
  //   - verify tokens: real token ids saved from previous round's sampling results.
  //   - placeholders: ptp_token_id_ slots whose sampling results will become next round's verify tokens.
  //
  // sampling_result_tokens layout: [result_0, ..., result_(k-1), result_k, ..., result_(k+p)]
  //   - result_0..result_(k-1): verification results corresponding to verify tokens.
  //   - result_k: the newly generated token (accepted output).
  //   - result_(k+1)..result_(k+p): sampling results at placeholder positions (saved for next round).
  //
  // Verification rules (checked sequentially for each verify token i in [0, verify_count)):
  //   1. Token match:   sampling_result_tokens[i] must equal draft_tokens[i], otherwise reject.
  //   2. Stop token:    if draft_tokens[i] is a stop token, stop verification (token itself is not accepted).
  //   3. Grammar check: if grammar_matcher is enabled, sampling_result_tokens[i+1] must satisfy
  //                     the grammar constraint, otherwise reject.
  //
  // After verification:
  //   - If all verify tokens accepted and placeholders exist: output result_k as generated token,
  //     save result_(k+1..k+p) as next round's verify tokens.
  //   - Otherwise: output result_(draft_hit_num) as generated token, discard all draft tokens.
  std::vector<int> draft_tokens = req->spec_state.draft_tokens.GetDraftTokens();
  size_t draft_hit_num = 0;

  for (size_t i = 0; i < verify_count; ++i, ++draft_hit_num) {
    if (req->sampling_result_tokens[i] != draft_tokens[i]) break;

    // Stop if stop token
    if (std::find(req->sampling_config.stop_token_ids.begin(), req->sampling_config.stop_token_ids.end(),
                  draft_tokens[i]) != req->sampling_config.stop_token_ids.end()) {
      break;
    }

    // Check grammar matcher constraint
    if (req->grammar_matcher) {
      if (!req->grammar_matcher->AcceptToken(req->sampling_result_tokens[i + 1])) {
        break;
      }
      req->grammar_matcher->Rollback(1);
    }
  }

  req->spec_state.accepted_tokens.assign(draft_tokens.begin(), draft_tokens.begin() + draft_hit_num);

  // Pop logprobs for unaccepted tokens
  size_t unaccepted_num = req->sampling_result_tokens.size() - kStepGenerateTokenNum - draft_hit_num;
  for (size_t i = 0; i < unaccepted_num; ++i) {
    if (!req->logprobs.empty()) req->logprobs.pop_back();
  }

  if (draft_hit_num == verify_count && ptp_placeholder_count > 0) {
    // All verify tokens accepted: take first placeholder result as generated, save the rest
    req->generated_tokens.clear();
    req->generated_tokens.push_back(req->sampling_result_tokens[verify_count]);

    req->spec_state.draft_tokens.ptp.clear();
    for (size_t i = verify_count + kStepGenerateTokenNum; i < req->sampling_result_tokens.size(); ++i) {
      req->spec_state.draft_tokens.ptp.push_back(req->sampling_result_tokens[i]);
    }
  } else {
    // TODO(zezhao): Currently discards all draft tokens on partial acceptance, since placeholder
    // results are sampled under a diverged prefix. Consider supporting selective reuse of
    // placeholder results (e.g., skipping the rejected position) as trie does in the future.
    req->generated_tokens.clear();
    req->generated_tokens.push_back(req->sampling_result_tokens[draft_hit_num]);
    req->spec_state.draft_tokens.clear();
  }

  req->sampling_result_tokens.clear();
  KLLM_LOG_DEBUG << "PTP FilterDraftTokens: req_id=" << req->req_id << ", hit=" << draft_hit_num << "/" << verify_count
                 << " (acceptance rate: " << (verify_count > 0 ? draft_hit_num * 100 / verify_count : 0) << "%)";
  return draft_hit_num;
}

}  // namespace ksana_llm
