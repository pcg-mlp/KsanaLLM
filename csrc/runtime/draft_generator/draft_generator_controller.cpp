/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/draft_generator/draft_generator_controller.h"
#include "fmt/format.h"
#include "profiler/reporter.h"
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
  if (hit_count.empty()) {
    return Status();
  }

  size_t total_hit_num = 0, total_draft_num = 0;
  for (size_t i = 0; i < hit_count.size(); ++i) {
    const auto& [hit_num, draft_num] = hit_count[i];
    total_hit_num += hit_num;
    total_draft_num += draft_num;
    KLLM_LOG_DEBUG << fmt::format("Draft index {} hit rate: {} / {} = {:.3f}%", i, hit_num, draft_num,
                                  static_cast<double>(hit_num) / static_cast<double>(draft_num) * 100.0);
  }

  // Accumulate per-position stats across calls so the periodic INFO log reflects
  // the entire report interval rather than a single batch.
  static std::vector<std::pair<size_t, size_t>> accumulated_hit_count;
  if (accumulated_hit_count.size() < hit_count.size()) {
    accumulated_hit_count.resize(hit_count.size());
  }
  for (size_t i = 0; i < hit_count.size(); ++i) {
    accumulated_hit_count[i].first += hit_count[i].first;
    accumulated_hit_count[i].second += hit_count[i].second;
  }

  // Aggregate INFO log every 10 seconds. hit_count being non-empty guarantees
  // total_draft_num > 0, so the divisions below are safe.
  constexpr size_t kReportIntervalMs = 10000;
  static IntervalLogger interval_logger(kReportIntervalMs);
  if (interval_logger.ShouldLog()) {
    size_t agg_hit = 0, agg_draft = 0;
    std::string per_index_str;
    for (const auto& [hit_num, draft_num] : accumulated_hit_count) {
      agg_hit += hit_num;
      agg_draft += draft_num;
      if (!per_index_str.empty()) per_index_str += ", ";
      per_index_str += fmt::format("{:.1f}%", draft_num > 0 ? hit_num * 100.f / draft_num : 0.f);
    }
    KLLM_LOG_INFO << fmt::format(
        "Draft hit rate: {:.1f}%, Hit tokens: {}, "
        "Draft tokens: {}; Per draft token hit rate: [{}]",
        agg_draft > 0 ? agg_hit * 100.f / agg_draft : 0.f, agg_hit, agg_draft, per_index_str);
    accumulated_hit_count.clear();
  }
  REPORT_METRIC("spec_draft_hit_num", total_hit_num);
  REPORT_METRIC("spec_draft_token_num", total_draft_num);

  return Status();
}

}  // namespace ksana_llm
