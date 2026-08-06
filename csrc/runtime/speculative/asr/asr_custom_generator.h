/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "runtime/speculative/common/draft_generator_interface.h"

namespace ksana_llm {

// Custom draft generator for ASR.
// It manages user-provided draft tokens (stored in req->draft_tokens.spec)
// across decode steps using a skip-ahead mechanism to handle minor mismatches.
class AsrCustomGenerator : public DraftGeneratorInterface {
 public:
  AsrCustomGenerator() = default;
  ~AsrCustomGenerator() override = default;

  void GenerateDraft(std::shared_ptr<ExecutorInferRequest> req) override;
};

}  // namespace ksana_llm
