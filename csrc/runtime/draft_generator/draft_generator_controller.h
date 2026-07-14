/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "runtime/draft_generator/draft_generator_interface.h"
#include "utils/status.h"

namespace ksana_llm {

// Manages multiple independent draft generators (e.g. TrieGenerator, PtpGenerator) and
// dispatches draft generation / filtering through them. Each generator is appended in order
// and runs independently during GenerateDraft; filtering delegates to the last generator.
class DraftGeneratorController {
 public:
  // Register a draft generator. Multiple generators can coexist (e.g. Trie + PTP).
  Status AppendDraftGenerator(std::shared_ptr<DraftGeneratorInterface> generator);

  // Run all registered generators to populate draft tokens for a single request.
  Status GenerateDraft(std::shared_ptr<ExecutorInferRequest> req);

  // Verify draft tokens against sampling results for a batch of requests,
  // collect per-position hit-rate statistics, and log them.
  Status FilterDraftTokens(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs);

 private:
  // Registered generators, executed in append order for generation.
  std::vector<std::shared_ptr<DraftGeneratorInterface>> draft_generators_;
};

}  // namespace ksana_llm