/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "runtime/draft_generator/draft_generator_controller.h"
#include "runtime/grammar_output.h"
#include "runtime/infer_request.h"
#include "runtime/schedule_output.h"
#include "runtime/structured_generation/structured_generator_factory.h"
#include "runtime/structured_generation/structured_generator_interface.h"

namespace ksana_llm {

class GenerationController {
 public:
  explicit GenerationController(std::shared_ptr<StructuredGeneratorFactory> structured_generator_factory)
      : structured_generator_factory_(structured_generator_factory) {}

  // Create structured generators for new requests (based on ScheduleOutput)
  // This method creates generators for requests in need_create_generator_req_ids and stores them persistently
  void CreateGenerators(std::shared_ptr<ScheduleOutput> schedule_output);

  // Restore generators from persistent storage for all requests in ScheduleOutput
  // This method assigns previously created generators back to requests
  void RestoreGenerators(std::shared_ptr<ScheduleOutput> schedule_output);

  // Remove generator for a finished request and clear the pointer in the request object
  void RemoveGenerator(int64_t req_id, std::vector<std::shared_ptr<ExecutorInferRequest>> &reqs);

  // Init generation state before sampling (for InferRequest vector)
  // Suggested to init before forwarding, so bitmask can be used in advance
  void InitGenerationState(std::vector<std::shared_ptr<InferRequest>> &reqs);

  // Update generation state after sampling_result_tokens are generated (for InferRequest)
  void UpdateGenerationState(std::vector<std::shared_ptr<ExecutorInferRequest>> &reqs);

  // Set draft generator controller for PTP (Parallel Token Prediction)
  void SetDraftGeneratorController(std::shared_ptr<DraftGeneratorController> draft_generator_controller);

  // Set structured generator factory
  void SetStructuredGeneratorFactory(std::shared_ptr<StructuredGeneratorFactory> structured_generator_factory);

  // Generate draft tokens for requests (PTP)
  Status GenerateDraft(std::shared_ptr<ExecutorInferRequest> req);

 private:
  void UpdateStructuredGeneratorState(std::vector<std::shared_ptr<ExecutorInferRequest>> &reqs);

  void FilterGeneratorDraftTokens(std::vector<std::shared_ptr<ExecutorInferRequest>> &reqs);

 private:
  std::shared_ptr<StructuredGeneratorFactory> structured_generator_factory_ = nullptr;

  // Draft generator controller for PTP (Parallel Token Prediction)
  std::shared_ptr<DraftGeneratorController> draft_generator_controller_ = nullptr;

  // Persistent storage for structured generators (keyed by request ID)
  // Generators are stored here to maintain state across multiple inference steps
  std::unordered_map<int64_t, std::shared_ptr<StructuredGeneratorInterface>> structured_generators_;

  // Mutex for thread-safe access to structured_generators_
  std::mutex generators_mutex_;
};

}  // namespace ksana_llm
