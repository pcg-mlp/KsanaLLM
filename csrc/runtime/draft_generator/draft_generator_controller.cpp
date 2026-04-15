/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/draft_generator/draft_generator_controller.h"
#include "fmt/format.h"
#include "runtime/executor_infer_request.h"
#include "utils/logger.h"

namespace ksana_llm {

Status DraftGeneratorController::AppendDraftGenerator(std::shared_ptr<DraftGeneratorInterface> generator) {
  draft_generators_.emplace_back(generator);
  return Status();
}

Status DraftGeneratorController::GenerateDraft(std::shared_ptr<ExecutorInferRequest> req) {
  for (auto& generator : draft_generators_) {
    generator->GenerateDraft(req);
  }
  return Status();
}

Status DraftGeneratorController::FilterDraftTokens(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs) {
  std::vector<std::pair<size_t, size_t>> hit_count;  // hit_num, draft_num

  for (auto& req : reqs) {
    // Save draft token count before filtering, as FilterDraftTokens may clear draft_tokens.
    const size_t draft_token_num = req->draft_tokens.size();

    size_t accepted_num = 0;
    if (!draft_generators_.empty()) {
      accepted_num = draft_generators_.back()->FilterDraftTokens(req);
    } else {
      accepted_num = DefaultFilterDraftTokens(req);
    }

    // Collect statistics
    if (draft_token_num > 0) {
      hit_count.resize(std::max(draft_token_num, hit_count.size()));
      for (size_t i = 0; i < draft_token_num; ++i) {
        auto& [hit_num, draft_num] = hit_count[i];
        ++draft_num;
        if (i < accepted_num) {
          ++hit_num;
        }
      }
    }
  }

  // Log statistics
  if (!hit_count.empty()) {
    size_t total_hit_num = 0, total_draft_num = 0;
    for (size_t i = 0; i < hit_count.size(); ++i) {
      const auto& [hit_num, draft_num] = hit_count[i];
      total_hit_num += hit_num;
      total_draft_num += draft_num;
      KLLM_LOG_DEBUG << fmt::format("Draft index {} hit rate: {} / {} = {:.3f}%", i, hit_num, draft_num,
                                    static_cast<double>(hit_num) / static_cast<double>(draft_num) * 100.0);
    }
  }

  return Status();
}

}  // namespace ksana_llm
