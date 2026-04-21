/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/generation_controller.h"

#include "profiler/reporter.h"
#include "utils/logger.h"

namespace ksana_llm {

void GenerationController::CreateGenerators(std::shared_ptr<ScheduleOutput> schedule_output) {
  if (!structured_generator_factory_) {
    KLLM_LOG_DEBUG << "Structured generator factory is null, skipping generator creation";
    return;
  }

  if (schedule_output->need_create_generator_req_ids.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(generators_mutex_);

  for (int64_t req_id : schedule_output->need_create_generator_req_ids) {
    KLLM_LOG_DEBUG << "Processing req_id: " << req_id << " for generator creation";

    // Find the corresponding request
    auto req_it =
        std::find_if(schedule_output->executor_running_reqs.begin(), schedule_output->executor_running_reqs.end(),
                     [req_id](const auto& req) { return req->req_id == req_id; });

    if (req_it == schedule_output->executor_running_reqs.end()) {
      KLLM_LOG_WARNING << "req_id " << req_id << " not found in executor_running_reqs";
      continue;
    }

    auto& req = *req_it;
    KLLM_LOG_DEBUG << "Found req_id: " << req_id
                   << ", HasConstraint: " << req->structured_generator_config.HasConstraint()
                   << ", constraint_type: " << static_cast<int>(req->structured_generator_config.constraint_type)
                   << ", constraint_spec: " << req->structured_generator_config.constraint_spec;

    if (!req->structured_generator_config.HasConstraint()) {
      continue;
    }

    try {
      auto generator = structured_generator_factory_->CreateGenerator(req->structured_generator_config,
                                                                      req->sampling_config.enable_thinking);
      if (!generator) {
        KLLM_LOG_WARNING << "Failed to create structured generator for req_id " << req_id;
        continue;
      }

      // Store in persistent map (will overwrite if exists)
      structured_generators_[req_id] = generator;
      // Also assign to current request
      req->structured_generator = generator;
      KLLM_LOG_INFO << "Created and stored structured generator for req_id: " << req_id;
    } catch (const std::exception& e) {
      KLLM_LOG_ERROR << "Failed to create structured generator for req_id " << req_id << ": " << e.what();
    }
  }
}

void GenerationController::RestoreGenerators(std::shared_ptr<ScheduleOutput> schedule_output) {
  if (!structured_generator_factory_) {
    return;
  }

  std::lock_guard<std::mutex> lock(generators_mutex_);

  for (auto& req : schedule_output->executor_running_reqs) {
    auto it = structured_generators_.find(req->req_id);
    if (it != structured_generators_.end()) {
      req->structured_generator = it->second;
      KLLM_LOG_DEBUG << "Restored generator for req_id: " << req->req_id;
    }
  }
}

void GenerationController::RemoveGenerator(int64_t req_id, std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs) {
  std::lock_guard<std::mutex> lock(generators_mutex_);

  auto it = structured_generators_.find(req_id);
  if (it != structured_generators_.end()) {
    structured_generators_.erase(it);
    KLLM_LOG_DEBUG << "Removed generator for req_id: " << req_id;
  }

  // Clear the pointer in the request object
  // This prevents FillNextTokenBitmask from being called on terminated generators during MTP
  for (auto& req : reqs) {
    if (req->req_id == req_id) {
      req->structured_generator = nullptr;
      KLLM_LOG_DEBUG << "Cleared structured_generator pointer for req_id: " << req_id;
      break;
    }
  }
}

void GenerationController::UpdateGenerationState(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs) {
  UpdateStructuredGeneratorState(reqs);

  // Clear stale draft_tokens left over from split-fuse intermediate prefill steps.
  for (auto& req : reqs) {
    if (req->forwarding_tokens_draft_num == 0) {
      req->draft_tokens.clear();
    }
  }

  // Filter draft tokens from both structured generation and PTP
  if (draft_generator_controller_) {
    draft_generator_controller_->FilterDraftTokens(reqs);
  } else {
    // Fallback to structured generator draft token filtering only
    FilterGeneratorDraftTokens(reqs);
  }
}

void GenerationController::SetDraftGeneratorController(
    std::shared_ptr<DraftGeneratorController> draft_generator_controller) {
  draft_generator_controller_ = draft_generator_controller;
}

void GenerationController::SetStructuredGeneratorFactory(
    std::shared_ptr<StructuredGeneratorFactory> structured_generator_factory) {
  structured_generator_factory_ = structured_generator_factory;
}

Status GenerationController::GenerateDraft(std::shared_ptr<ExecutorInferRequest> req) {
  if (draft_generator_controller_) {
    return draft_generator_controller_->GenerateDraft(req);
  }
  return Status();
}

// Update structured generator state after sampling
void GenerationController::UpdateStructuredGeneratorState(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs) {
  for (auto& req : reqs) {
    if (!req->structured_generator || req->sampling_result_tokens.empty()) {
      continue;
    }

    if (req->structured_generator->IsTerminated()) {
      KLLM_LOG_DEBUG << "Structured generation completed for request " << req->req_id;
      continue;
    }
    int token_id = req->sampling_result_tokens.front();
    bool accepted = req->structured_generator->AcceptToken(token_id);
    if (!accepted) {
      KLLM_LOG_WARNING << "Structured generator rejected token " << token_id << " for req_id " << req->req_id;
    }
  }
}

void GenerationController::FilterGeneratorDraftTokens(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs) {
  std::vector<std::pair<size_t, size_t>> hit_count;  // hit_num, draft_num
  for (auto& req : reqs) {
    req->accepted_tokens.clear();
    if (req->sampling_result_tokens.size() - kStepGenerateTokenNum != req->draft_tokens.size() &&
        req->logits_custom_length == 0) {
      KLLM_LOG_ERROR << fmt::format(
          "req {} sampling_result_tokens.size = {}, mtp_draft_tokens.size = {}, spec_draft_tokens.size = {}",
          req->req_id, req->sampling_result_tokens.size(), req->draft_tokens.mtp.size(), req->draft_tokens.spec.size());
      continue;
    }
    // Check which tokens are predicted correctly.
    size_t draft_hit_num = 0;
    std::vector<int> draft_tokens = req->draft_tokens.GetDraftTokens();
    for (size_t i = 0; i < draft_tokens.size(); ++i, ++draft_hit_num) {
      if (req->sampling_result_tokens[i] != draft_tokens[i]) {
        break;
      }
      // Structured constraint check for the new generated token
      if (req->structured_generator) {
        if (req->structured_generator->IsTerminated()) {
          KLLM_LOG_DEBUG << "Structured generator terminated for req_id " << req->req_id
                         << ", stopping draft token acceptance";
          break;
        }

        const int new_token = req->sampling_result_tokens[i + 1];
        if (!req->structured_generator->AcceptToken(new_token)) {
          KLLM_LOG_DEBUG << "Structured constraint rejected new_token " << new_token << " for request " << req->req_id
                         << ", will not use it as generated_token";
          break;
        }
      }

      // stop if stop token
      if (std::find(req->sampling_config.stop_token_ids.begin(), req->sampling_config.stop_token_ids.end(),
                    draft_tokens[i]) != req->sampling_config.stop_token_ids.end()) {
        break;
      }
    }

    hit_count.resize(std::max(req->draft_tokens.size(), hit_count.size()));
    for (size_t i = 0; i < req->draft_tokens.size(); ++i) {
      auto& [hit_num, draft_num] = hit_count[i];
      ++draft_num;
      if (i < draft_hit_num) {
        ++hit_num;
      }
    }

    req->accepted_tokens.swap(draft_tokens);
    req->accepted_tokens.resize(draft_hit_num);
    // Delete logprobs of tokens that were not accepted.
    size_t unaccepted_num =
        draft_tokens.size() > 0 ? req->sampling_result_tokens.size() - kStepGenerateTokenNum - draft_hit_num : 0;
    for (size_t i = 0; i < unaccepted_num; ++i) {
      if (!req->logprobs.empty()) req->logprobs.pop_back();
    }
    // replace generated_token with draft token's next token
    req->generated_tokens.clear();
    req->generated_tokens.push_back(req->sampling_result_tokens[draft_hit_num]);
    req->sampling_result_tokens.clear();
  }

  if (hit_count.empty()) {
    return;
  }

  size_t total_hit_num = 0, total_draft_num = 0;
  for (size_t i = 0; i < hit_count.size(); ++i) {
    const auto& [hit_num, draft_num] = hit_count[i];
    KLLM_LOG_DEBUG << fmt::format("Draft index {} hit rate: {} / {}", i, hit_num, draft_num);
    total_hit_num += hit_num;
    total_draft_num += draft_num;
  }
  // Log every 10 seconds
  constexpr size_t kReportIntervalMs = 10000;
  static IntervalLogger interval_logger(kReportIntervalMs);
  if (interval_logger.ShouldLog()) {
    KLLM_LOG_INFO << fmt::format(
        "Draft hit rate: {:.1f}%, Mean acceptance length: {:.2f}, Hit tokens: {}, Draft tokens: {}",
        total_hit_num * 100.f / total_draft_num, 1 + total_hit_num * 1.f / reqs.size(), total_hit_num, total_draft_num);
  }
  // Report
  REPORT_METRIC("spec_draft_hit_num", total_hit_num);
  REPORT_METRIC("spec_draft_token_num", total_draft_num);
}

}  // namespace ksana_llm
