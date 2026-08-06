/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

namespace ksana_llm {

// Forward declaration
class ExecutorInferRequest;

// RAII guard that saves and restores per-request state modified during MTP forward.
// Saves 4 fields on construction: forwarding_tokens, kv_cached_token_num,
// prefix_cache_len, forwarding_tokens_draft_num.
// Restores all 4 fields on destruction (noexcept) — guaranteeing recovery even if
// an exception is thrown during the MTP forward loop.
//
// Usage:
//   MtpStateGuard guard(reqs);  // saves state
//   // ... MTP forward loop modifies reqs ...
//   // guard destructor restores state automatically
class MtpStateGuard {
 public:
  // Save the current state of each request in `reqs`.
  // The saved state is keyed by req_id for O(1) lookup during restore.
  explicit MtpStateGuard(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs);

  // Restore all saved state. noexcept guarantees this runs during stack unwinding.
  ~MtpStateGuard() noexcept;

  // Non-copyable, non-movable.
  MtpStateGuard(const MtpStateGuard&) = delete;
  MtpStateGuard& operator=(const MtpStateGuard&) = delete;
  MtpStateGuard(MtpStateGuard&&) = delete;
  MtpStateGuard& operator=(MtpStateGuard&&) = delete;

 private:
  struct SavedState {
    std::vector<int> forwarding_tokens;
    size_t kv_cached_token_num = 0;
    size_t prefix_cache_len = 0;
    size_t forwarding_tokens_draft_num = 0;
  };

  // req_id -> saved state
  std::unordered_map<int64_t, SavedState> saved_states_;
  // Pointer to the reqs vector for restoration (non-owning)
  std::vector<std::shared_ptr<ExecutorInferRequest>>* reqs_;
};

}  // namespace ksana_llm