/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "runtime/speculative/common/draft_generator_interface.h"
#include "runtime/speculative/common/spec_scheduling_info.h"
#include "utils/status.h"

namespace ksana_llm {

// Forward declarations
class ModelExecutor;
class ModelSampler;
class ModelBuilder;
class ModelInstance;
class ScheduleOutput;
class ExecutorInferRequest;
class HiddenUnitDeviceBuffer;
class Waiter;

// DSpark speculative decoder.
// Encapsulates the DSpark forward + per-step head-only sampling loop.
// Implements DraftGeneratorInterface, with GenerateDraft as a no-op —
// draft tokens are produced in ExecuteForward.
//
// Owns the prev_dspark_done_ Waiter used to serialize main forward against DSpark forward
// on PP > 1 ranks to avoid collective operation deadlocks.
class DSparkDecoder : public DraftGeneratorInterface {
 public:
  // Constructor.
  // - dspark_block_size: number of DSpark draft tokens per step
  // - dspark_noise_token_id: noise token ID for padding
  // - infer_req_mutex: ExecutorRuntime::infer_req_mutex_, shared with UpdateCachedRequests
  //   (Preprocess thread) to prevent it from racing with ExecuteForward's forward loop over the
  //   same ExecutorInferRequest fields (non-owning, lifetime > DSparkDecoder).
  // - serialize_dspark: true if main forward must wait for DSpark completion (PP > 1, sampling node).
  DSparkDecoder(size_t dspark_block_size, int dspark_noise_token_id, std::mutex* infer_req_mutex,
                bool serialize_dspark);

  // DraftGeneratorInterface implementation.
  void GenerateDraft(std::shared_ptr<ExecutorInferRequest>) override {}
  SpecSchedulingInfo GetSchedulingInfo() const override;

  // Static helper: compute scheduling info from block_size.
  // Single source of truth shared by GetSchedulingInfo() and DSparkEstimateProfile.
  static SpecSchedulingInfo ComputeSchedulingInfo(size_t block_size);

  // Execute the DSpark forward + head-only sampling loop for all requests.
  Status ExecuteForward(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs, HiddenUnitDeviceBuffer* hidden_unit,
                        std::shared_ptr<ScheduleOutput> schedule_output, ModelExecutor* model_executor,
                        ModelSampler* model_sampler, ModelBuilder* model_builder, ModelInstance* model_instance);

  // Shutdown: wake up any thread waiting on prev_dspark_done_.
  void Shutdown();

  // Get the Waiter shared with ExecutorRuntime::Forward() for main/DSpark serialization.
  std::shared_ptr<Waiter> GetPrevDsparkDoneWaiter() const { return prev_dspark_done_; }

  // Whether main-DSpark serialization is required (PP > 1). Used by DSparkDecoderAdapter::PreMainForward.
  bool IsSerializationRequired() const { return serialize_dspark_; }

 private:
  size_t dspark_block_size_;
  int dspark_noise_token_id_;
  bool serialize_dspark_;

  // Serialize main forward against DSpark forward on PP > 1 sampling ranks to avoid TrtAllReduce deadlock.
  std::shared_ptr<Waiter> prev_dspark_done_;

  // Protect concurrent modification of infer_request by DSpark forward and UpdateInferRequest.
  // Non-owning: must be ExecutorRuntime::infer_req_mutex_, the same instance UpdateCachedRequests
  // locks, otherwise the two threads do not actually mutually exclude each other.
  std::mutex* infer_req_mutex_;
};

}  // namespace ksana_llm
