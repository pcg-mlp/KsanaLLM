/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

namespace ksana_llm {

// Forward declaration
class ExecutorInferRequest;

class DraftGeneratorInterface {
 public:
  virtual ~DraftGeneratorInterface() {}

  // Generate draft tokens for the request
  virtual void GenerateDraft(std::shared_ptr<ExecutorInferRequest> req) = 0;

  // Filter and verify draft tokens for a single request
  // Returns the number of accepted draft tokens
  // Default implementation uses the standard verification logic
  virtual size_t FilterDraftTokens(std::shared_ptr<ExecutorInferRequest> req);
};

// Default draft token filtering logic used by MTP and Trie generators.
//
// Precondition:
//   sampling_result_tokens.size() - kStepGenerateTokenNum == draft_tokens.size()
//
// Side effects on req:
//   - accepted_tokens: set to the prefix of draft tokens that matched sampling results.
//   - generated_tokens: set to the first unmatched sampling result token.
//   - sampling_result_tokens: cleared.
//   - logprobs: trailing entries for unaccepted tokens are removed.
//
// Returns the number of accepted (matched) draft tokens.
size_t DefaultFilterDraftTokens(std::shared_ptr<ExecutorInferRequest> req);

}  // namespace ksana_llm