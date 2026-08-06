/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/speculative/mtp/mtp_state_guard.h"

#include "runtime/executor_infer_request.h"
#include "utils/logger.h"

namespace ksana_llm {

MtpStateGuard::MtpStateGuard(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs) : reqs_(&reqs) {
  saved_states_.reserve(reqs.size());
  for (const auto& req : reqs) {
    SavedState state;
    state.forwarding_tokens = req->forwarding_tokens;
    state.kv_cached_token_num = req->kv_cached_token_num;
    state.prefix_cache_len = req->prefix_cache_len;
    state.forwarding_tokens_draft_num = req->spec_state.forwarding_tokens_draft_num;
    saved_states_.emplace(req->req_id, std::move(state));
  }
  KLLM_LOG_DEBUG << "MtpStateGuard saved state for " << saved_states_.size() << " requests";
}

MtpStateGuard::~MtpStateGuard() noexcept {
  if (!reqs_) return;
  for (auto& req : *reqs_) {
    auto it = saved_states_.find(req->req_id);
    if (it == saved_states_.end()) {
      KLLM_LOG_ERROR << "MtpStateGuard: req_id " << req->req_id << " not found in saved states";
      continue;
    }
    const auto& saved = it->second;
    req->forwarding_tokens = saved.forwarding_tokens;
    req->kv_cached_token_num = saved.kv_cached_token_num;
    req->prefix_cache_len = saved.prefix_cache_len;
    req->spec_state.forwarding_tokens_draft_num = saved.forwarding_tokens_draft_num;
  }
  KLLM_LOG_DEBUG << "MtpStateGuard restored state for " << saved_states_.size() << " requests";
}

}  // namespace ksana_llm