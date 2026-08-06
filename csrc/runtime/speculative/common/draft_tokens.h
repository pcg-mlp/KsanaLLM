/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cassert>
#include <vector>

namespace ksana_llm {

struct DraftTokens {
  std::vector<int> mtp;
  std::vector<int> spec;
  std::vector<int> ptp;

  std::vector<int> GetDraftTokens() const {
    std::vector<int> draft_tokens;
    draft_tokens.reserve(mtp.size() + spec.size() + ptp.size());
    draft_tokens.insert(draft_tokens.end(), mtp.begin(), mtp.end());
    draft_tokens.insert(draft_tokens.end(), spec.begin(), spec.end());
    draft_tokens.insert(draft_tokens.end(), ptp.begin(), ptp.end());
    return draft_tokens;
  }

  void TruncDraft(size_t new_size) {
    assert(new_size <= mtp.size() + spec.size() + ptp.size());
    if (mtp.size() > new_size) {
      mtp.resize(new_size);
      spec.clear();
      ptp.clear();
      return;
    }
    new_size -= mtp.size();
    if (spec.size() > new_size) {
      spec.resize(new_size);
      ptp.clear();
      return;
    }
    new_size -= spec.size();
    if (ptp.size() > new_size) {
      ptp.resize(new_size);
    }
  }

  void clear() {
    mtp.clear();
    spec.clear();
    ptp.clear();
  }

  size_t size() const { return mtp.size() + spec.size() + ptp.size(); }
};

// Combines generated_tokens + draft_tokens from a request type.
template <typename RequestT>
std::vector<int> GetAllProducedTokens(const RequestT& req) {
  std::vector<int> all;
  const auto& draft = req.spec_state.draft_tokens;
  all.reserve(req.generated_tokens.size() + draft.size());
  all.insert(all.end(), req.generated_tokens.begin(), req.generated_tokens.end());
  auto drafts = draft.GetDraftTokens();
  all.insert(all.end(), drafts.begin(), drafts.end());
  return all;
}

}  // namespace ksana_llm
