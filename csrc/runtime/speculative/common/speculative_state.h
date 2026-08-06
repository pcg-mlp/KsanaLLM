/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <vector>

#include "runtime/speculative/common/draft_tokens.h"

namespace ksana_llm {

// Speculative decoding state owned by each ExecutorInferRequest.
// Isolated from the main inference flow to provide clear ownership
// boundaries between speculative and main execution paths.
struct SpeculativeState {
  DraftTokens draft_tokens;

  // Draft tokens that were accepted by the main model verification.
  std::vector<int> accepted_tokens;

  // Number of draft tokens appended to forwarding_tokens.
  // Tracked to correctly undo speculative padding during state restore.
  size_t forwarding_tokens_draft_num = 0;

  size_t GetDraftTokenCount() const { return draft_tokens.size(); }
  void ClearDraftTokens() { draft_tokens.clear(); }
};

}  // namespace ksana_llm
