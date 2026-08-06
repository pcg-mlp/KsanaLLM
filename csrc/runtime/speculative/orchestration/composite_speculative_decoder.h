/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "runtime/speculative/common/speculative_decoder_interface.h"
#include "utils/status.h"

namespace ksana_llm {

// Forward declarations
class GenerationController;

// Manages multiple SpeculativeDecoderInterface instances.
// Provides registration, execution orchestration, scheduling info aggregation,
// and compatibility checking.
//
// Registration order determines draft token concatenation order (mtp → spec → ptp),
// which must match DraftTokens::GetDraftTokens() ordering.
class CompositeSpeculativeDecoder {
 public:
  CompositeSpeculativeDecoder() = default;
  ~CompositeSpeculativeDecoder() = default;

  // Register a decoder. Order matters — first registered = first in draft token sequence.
  void Register(std::unique_ptr<SpeculativeDecoderInterface> decoder);

  // Aggregate scheduling info from all registered decoders.
  SpecSchedulingInfo GetAggregatedSchedulingInfo() const;

  // Execute BeforeForward on all decoders in registration order.
  void BeforeForward(SpecContext& ctx);

  // Execute ExecuteForward on all decoders in registration order.
  // No-draft-model decoders are no-ops; draft-model decoders execute their forward pass.
  void ExecuteForward(SpecContext& ctx);

  // Execute AfterForward on all decoders in registration order.
  void AfterForward(SpecContext& ctx);

  // Execute Verify on all decoders in registration order.
  void Verify(SpecContext& ctx);

  // Verify draft tokens for all requests using the default verification logic.
  // This is the common case when no decoder-specific verification is needed.
  void VerifyDraftTokens(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs);

  // Check whether all registered decoders are pairwise compatible.
  bool AreAllCompatible() const;

  // Decoupling hooks.
  // Called before the main model forward (ExecutorRuntime::Forward).
  void PreMainForward(SpecContext& ctx);

  // Called during ExecutorRuntime::Shutdown() to wake blocked threads.
  void Shutdown();

  // Encapsulates post-processing: UpdateGenerationState → VerifyDraftTokens →
  // BeforeForward → ExecuteForward → AfterForward into a single call.
  // When no decoders are registered, only UpdateGenerationState and default draft
  // verification run (Steps 1-2); the speculative forward steps (3-4) are skipped.
  // - running_reqs: model_output->schedule_output->executor_running_reqs
  // - hidden_unit: model_output->hidden_unit (for ExecuteForward)
  // - schedule_output: model_output->schedule_output
  // - generation_controller: for UpdateGenerationState delegation (nullable)
  void HandlePostprocess(std::vector<std::shared_ptr<ExecutorInferRequest>>& running_reqs,
                         HiddenUnitDeviceBuffer* hidden_unit, std::shared_ptr<ScheduleOutput> schedule_output,
                         GenerationController* generation_controller);

  // Dependency injection for GenerationController delegation.
  void SetGenerationController(GenerationController* gc) { generation_controller_ = gc; }

  // Number of registered decoders.
  size_t GetDecoderCount() const { return decoders_.size(); }

  // Access to individual decoders (for testing).
  const std::vector<std::unique_ptr<SpeculativeDecoderInterface>>& GetDecoders() const { return decoders_; }

 private:
  // Collect per-position draft token hit-rate statistics and log aggregate every 10 seconds.
  static void CollectAndLogDraftHitStats(const std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs,
                                         const std::vector<size_t>& pre_filter_draft_counts);

  std::vector<std::unique_ptr<SpeculativeDecoderInterface>> decoders_;
  GenerationController* generation_controller_ = nullptr;  // non-owning
};

}  // namespace ksana_llm