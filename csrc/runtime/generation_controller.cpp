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
      KLLM_LOG_DEBUG << "Created and stored structured generator for req_id: " << req_id;
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

void GenerationController::RemoveGenerator(std::shared_ptr<ExecutorInferRequest>& req) {
  std::lock_guard<std::mutex> lock(generators_mutex_);

  if (structured_generators_.erase(req->req_id) > 0) {
    KLLM_LOG_DEBUG << "Removed generator for req_id: " << req->req_id;
  }

  // Clear the pointer in the request object
  // This prevents FillNextTokenBitmask from being called on terminated generators during MTP
  req->structured_generator = nullptr;
  KLLM_LOG_DEBUG << "Cleared structured_generator pointer for req_id: " << req->req_id;
}

void GenerationController::UpdateGenerationState(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs) {
  UpdateStructuredGeneratorState(reqs);
  // Stale draft_tokens cleanup has been moved to CompositeSpeculativeDecoder::HandlePostprocess.
}

void GenerationController::SetStructuredGeneratorFactory(
    std::shared_ptr<StructuredGeneratorFactory> structured_generator_factory) {
  structured_generator_factory_ = structured_generator_factory;
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

}  // namespace ksana_llm
