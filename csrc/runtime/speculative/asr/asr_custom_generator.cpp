/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/speculative/asr/asr_custom_generator.h"

#include "runtime/executor_infer_request.h"
#include "utils/string_utils.h"

namespace ksana_llm {

void AsrCustomGenerator::GenerateDraft(std::shared_ptr<ExecutorInferRequest> req) {
  const size_t unverified_token_num = req->spec_state.draft_tokens.mtp.size();
  size_t accepted_tokens = req->spec_state.accepted_tokens.size();
  // Already mismatch in MTP part
  if (accepted_tokens < unverified_token_num) {
    return;
  }
  accepted_tokens -= unverified_token_num;

  std::vector<int>& draft_tokens = req->spec_state.draft_tokens.spec;
  if (accepted_tokens >= draft_tokens.size()) {  // Full match
    draft_tokens.clear();
  } else {
    KLLM_CHECK(req->generated_tokens.size() == 1);
    const int generated_token = req->generated_tokens.front();
    KLLM_CHECK_WITH_INFO(draft_tokens[accepted_tokens] != generated_token,
                         fmt::format("Expected `draft_tokens[accepted_tokens]`={} to mismatch `generated_token`={}",
                                     draft_tokens[accepted_tokens], generated_token));

    // Default: discard from the mismatch point onward.
    size_t next_draft_tokens_pos = accepted_tokens;
    // Try to find a match for `generated_token` within the next few draft tokens.
    // If found, skip the mismatched tokens and reuse the rest of the draft.
    constexpr size_t kMaxSkipDraftTokenNum = 3;
    for (size_t i = 1; i <= kMaxSkipDraftTokenNum; i++) {
      if (accepted_tokens + i < draft_tokens.size() && draft_tokens[accepted_tokens + i] == generated_token) {
        next_draft_tokens_pos = accepted_tokens + i + 1;  // Reuse tokens after the matched position.
        break;
      }
    }

    // Extract the remaining reusable draft tokens and replace the old draft.
    draft_tokens.erase(draft_tokens.begin(), draft_tokens.begin() + next_draft_tokens_pos);
  }

  KLLM_LOG_DEBUG << "Generated draft tokens " << Vector2Str(draft_tokens) << " for request " << req->req_id;
}

}  // namespace ksana_llm
