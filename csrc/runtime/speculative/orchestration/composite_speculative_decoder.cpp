/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/speculative/orchestration/composite_speculative_decoder.h"

#include <string>

#include "runtime/executor_infer_request.h"
#include "runtime/generation_controller.h"
#include "runtime/speculative/common/draft_generator_interface.h"
#include "utils/logger.h"

namespace ksana_llm {

void CompositeSpeculativeDecoder::Register(std::unique_ptr<SpeculativeDecoderInterface> decoder) {
  if (!decoder) {
    KLLM_LOG_WARNING << "CompositeSpeculativeDecoder::Register called with null decoder";
    return;
  }
  KLLM_LOG_INFO << "Registered decoder: " << decoder->GetName();
  decoders_.push_back(std::move(decoder));
}

SpecSchedulingInfo CompositeSpeculativeDecoder::GetAggregatedSchedulingInfo() const {
  SpecSchedulingInfo aggregated;
  for (const auto& decoder : decoders_) {
    auto info = decoder->GetSchedulingInfo();
    aggregated.extra_kv_tokens += info.extra_kv_tokens;
    aggregated.extra_token_capacity += info.extra_token_capacity;
    aggregated.workload_multiplier += info.workload_multiplier;
  }
  return aggregated;
}

void CompositeSpeculativeDecoder::BeforeForward(SpecContext& ctx) {
  for (auto& decoder : decoders_) {
    decoder->BeforeForward(ctx);
  }
}

void CompositeSpeculativeDecoder::ExecuteForward(SpecContext& ctx) {
  for (auto& decoder : decoders_) {
    decoder->ExecuteForward(ctx);
  }
}

void CompositeSpeculativeDecoder::AfterForward(SpecContext& ctx) {
  for (auto& decoder : decoders_) {
    decoder->AfterForward(ctx);
  }
}

void CompositeSpeculativeDecoder::Verify(SpecContext& ctx) {
  for (auto& decoder : decoders_) {
    decoder->Verify(ctx);
  }
}

void CompositeSpeculativeDecoder::PreMainForward(SpecContext& ctx) {
  for (auto& decoder : decoders_) {
    decoder->PreMainForward(ctx);
  }
}

void CompositeSpeculativeDecoder::Shutdown() {
  for (auto& decoder : decoders_) {
    decoder->Shutdown();
  }
}

void CompositeSpeculativeDecoder::HandlePostprocess(std::vector<std::shared_ptr<ExecutorInferRequest>>& running_reqs,
                                                    HiddenUnitDeviceBuffer* hidden_unit,
                                                    std::shared_ptr<ScheduleOutput> schedule_output,
                                                    GenerationController* generation_controller) {
  // Step 1: UpdateGenerationState clears stale structured generator state.
  if (generation_controller) {
    generation_controller->UpdateGenerationState(running_reqs);
  }

  // Clear stale draft_tokens left over from split-fuse intermediate prefill steps.
  // Must run BEFORE VerifyDraftTokens (RCA 2026-07-21).
  for (auto& req : running_reqs) {
    if (req->spec_state.forwarding_tokens_draft_num == 0) {
      req->spec_state.draft_tokens.clear();
    }
  }

  // Step 2: Verify draft tokens against sampling results. This also moves
  // sampling_result_tokens into generated_tokens even when no speculative
  // decoder is registered (see DefaultFilterDraftTokens fallback below) —
  // it must always run, regardless of decoders_.empty().
  VerifyDraftTokens(running_reqs);

  // The remaining steps (generate draft tokens for the next decode step, and
  // execute the speculative forward pass) only apply when decoders are registered.
  if (decoders_.empty()) return;

  // Step 3: Generate draft tokens for the next decode step.
  {
    // Decoders write draft tokens directly into req->spec_state.draft_tokens;
    // the ctx.output field in ForDraftGen is unused (pass nullptr).
    SpecContext ctx = SpecContext::ForDraftGen(&running_reqs, nullptr);
    for (auto& req : *ctx.reqs) {
      KLLM_LOG_DEBUG << "Before BeforeForward: req_id=" << req->req_id
                     << ", draft_tokens.size=" << req->spec_state.draft_tokens.size();
    }
    BeforeForward(ctx);
    for (auto& req : *ctx.reqs) {
      KLLM_LOG_DEBUG << "After BeforeForward: req_id=" << req->req_id
                     << ", draft_tokens.size=" << req->spec_state.draft_tokens.size()
                     << ", spec=" << req->spec_state.draft_tokens.spec.size();
    }
  }

  // Step 4: Update sampling_token_num for scheduling/workload estimation.
  for (auto& req : running_reqs) {
    size_t draft_token_num = req->spec_state.draft_tokens.size();
    req->sampling_token_num = kStepGenerateTokenNum + draft_token_num;
  }

  // Step 5: Execute speculative forward pass (MTP/DSpark decode).
  // hidden_unit shape and dtype are set by the caller (ModelRunner::HandlePostprocess).
  {
    auto ctx = SpecContext::ForMtp(&running_reqs, hidden_unit, schedule_output, nullptr);
    ExecuteForward(ctx);
    AfterForward(ctx);
  }
}

void CompositeSpeculativeDecoder::VerifyDraftTokens(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs) {
  KLLM_LOG_DEBUG << "VerifyDraftTokens: decoders=" << decoders_.size() << ", reqs=" << reqs.size();

  // Save draft token count before filtering (some filter implementations may clear draft_tokens).
  std::vector<size_t> pre_filter_draft_counts;
  pre_filter_draft_counts.reserve(reqs.size());
  for (const auto& req : reqs) {
    pre_filter_draft_counts.push_back(req->spec_state.draft_tokens.size());
  }

  // If specific decoders are registered, delegate to their Verify().
  if (!decoders_.empty()) {
    for (auto& req : reqs) {
      for (auto& decoder : decoders_) {
        // Build a single-request context for each decoder.
        std::vector<std::shared_ptr<ExecutorInferRequest>> single_req = {req};
        auto ctx = SpecContext::ForVerify(&single_req);
        decoder->Verify(ctx);
      }
    }
  } else {
    // Fallback: use the default verification logic for all requests.
    for (auto& req : reqs) {
      DefaultFilterDraftTokens(req);
    }
  }

  // Collect per-position hit-rate statistics and log aggregate every 10 seconds.
  CollectAndLogDraftHitStats(reqs, pre_filter_draft_counts);
}

bool CompositeSpeculativeDecoder::AreAllCompatible() const {
  for (size_t i = 0; i < decoders_.size(); ++i) {
    for (size_t j = i + 1; j < decoders_.size(); ++j) {
      if (!decoders_[i]->IsCombinableWith(decoders_[j].get())) {
        KLLM_LOG_WARNING << "Decoders " << decoders_[i]->GetName() << " and " << decoders_[j]->GetName()
                         << " are incompatible";
        return false;
      }
    }
  }
  return true;
}

void CompositeSpeculativeDecoder::CollectAndLogDraftHitStats(
    const std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs,
    const std::vector<size_t>& pre_filter_draft_counts) {
  // Accumulate per-position hit/draft counts across all requests in this batch.
  static std::vector<std::pair<size_t, size_t>> accumulated;
  for (size_t i = 0; i < reqs.size(); ++i) {
    const size_t draft_num = pre_filter_draft_counts[i];
    if (draft_num == 0) continue;
    const size_t hit_num = reqs[i]->spec_state.accepted_tokens.size();
    if (accumulated.size() < draft_num) accumulated.resize(draft_num);
    for (size_t pos = 0; pos < draft_num; ++pos) {
      accumulated[pos].first += (pos < hit_num ? 1 : 0);
      accumulated[pos].second += 1;
    }
  }

  // Aggregate INFO log every 10 seconds.
  constexpr size_t kReportIntervalMs = 10000;
  static IntervalLogger interval_logger(kReportIntervalMs);
  if (interval_logger.ShouldLog()) {
    size_t agg_hit = 0, agg_draft = 0;
    std::string per_index_str;
    for (const auto& [hit, draft] : accumulated) {
      agg_hit += hit;
      agg_draft += draft;
      if (!per_index_str.empty()) per_index_str += ", ";
      per_index_str += fmt::format("{:.1f}%", draft > 0 ? hit * 100.f / draft : 0.f);
    }
    KLLM_LOG_INFO << fmt::format(
        "Draft hit rate: {:.1f}%, Hit tokens: {}, Draft tokens: {}; Per draft token hit rate: [{}]",
        agg_draft > 0 ? agg_hit * 100.f / agg_draft : 0.f, agg_hit, agg_draft, per_index_str);
    accumulated.clear();
  }
}

}  // namespace ksana_llm