/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <mutex>

#include "runtime/speculative/common/draft_generator_interface.h"
#include "runtime/speculative/common/spec_scheduling_info.h"
#include "utils/status.h"

namespace ksana_llm {

// Forward declarations
class DraftModelInterface;
class ModelDelegate;
class ModelExecutor;
class ModelSampler;
class ModelBuilder;
class ScheduleOutput;
class ExecutorInferRequest;
class HiddenUnitDeviceBuffer;
class Waiter;
class MtpGatherIndexBuilder;
class ForwardingContext;

// MTP (Multi-Token Prediction) decoder implementation.
// Encapsulates the MTP forward loop: state save/restore, step iteration,
// BuildModelInput → DraftModelInterface::Forward() → Sampling, draft token collection.
//
// Owns the prev_mtp_done_ Waiter used to serialize main forward against MTP forward
// on TP > 1 ranks to avoid collective operation deadlocks.
class MtpDecoder : public DraftGeneratorInterface {
 public:
  // Constructor.
  // - draft_model: the MTP draft model execution engine (non-owning, lifetime > MtpDecoder)
  // - mtp_step_num: number of MTP draft tokens to generate per step
  // - serialize_main_mtp: true if main forward must wait for MTP completion (TP > 1)
  // - model_delegate: narrow interface to CommonModel operations (non-owning)
  // - infer_req_mutex: ExecutorRuntime::infer_req_mutex_, shared with UpdateCachedRequests
  //   (Preprocess thread) to prevent it from racing with this class's multi-step forward loop
  //   over the same ExecutorInferRequest fields (non-owning, lifetime > MtpDecoder).
  MtpDecoder(DraftModelInterface* draft_model, size_t mtp_step_num, bool serialize_main_mtp,
             ModelDelegate* model_delegate, std::mutex* infer_req_mutex);

  ~MtpDecoder() override;

  // DraftGeneratorInterface implementation.
  // GenerateDraft is a no-op for MTP — draft tokens are produced in ExecuteForward().
  void GenerateDraft(std::shared_ptr<ExecutorInferRequest> req) override {}

  // Return MTP scheduling resource requirements.
  SpecSchedulingInfo GetSchedulingInfo() const override;

  // Static helper: compute scheduling info from step_num.
  // Single source of truth shared by GetSchedulingInfo() and MtpEstimateProfile.
  static SpecSchedulingInfo ComputeSchedulingInfo(size_t step_num);

  // Execute the MTP forward loop for all requests.
  // - reqs: requests to run MTP on
  // - hidden_unit: hidden state buffer (shared with main forward)
  // - schedule_output: schedule output for BuildModelInput
  // - model_executor: for weight access only (non-owning)
  // - model_sampler: for Sampling() calls (non-owning)
  // - model_builder: for BuildModelInput() calls (non-owning)
  // CommonModel operations are accessed through model_delegate_ (injected at construction).
  Status ExecuteForward(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs, HiddenUnitDeviceBuffer* hidden_unit,
                        std::shared_ptr<ScheduleOutput> schedule_output, ModelExecutor* model_executor,
                        ModelSampler* model_sampler, ModelBuilder* model_builder);

  // Shutdown: wake up any thread waiting on prev_mtp_done_.
  void Shutdown();

  // Get the Waiter shared with ExecutorRuntime::Forward() for main/MTP serialization.
  std::shared_ptr<Waiter> GetPrevMtpDoneWaiter() const { return prev_mtp_done_; }

  // Whether main-MTP serialization is required (TP > 1). Used by MtpDecoderAdapter::PreMainForward.
  bool IsSerializationRequired() const { return serialize_main_mtp_; }

 public:
  // Setup gather index builder override on the forwarding context before
  // the main forward runs. Called by ExecutorRuntime or ModelRunner before Forward().
  // - fwd_ctx: the forwarding context (shared between main and MTP forward)
  void PrepareForMainForward(ForwardingContext& fwd_ctx);

  // Cleanup after MTP forward to reset external overrides.
  void CleanupAfterMtpForward(ForwardingContext& fwd_ctx);

 private:
  DraftModelInterface* draft_model_;  // Non-owning
  ModelDelegate* model_delegate_;     // Non-owning
  size_t mtp_step_num_;
  bool serialize_main_mtp_;

  // MTP-specific gather index builder. Created lazily on first use.
  // Injected into ModelInput via SetExternalGatherIdxBuilder before MTP forward.
  std::unique_ptr<MtpGatherIndexBuilder> gather_idx_builder_;

  // Serialize main forward against MTP forward on TP > 1 ranks to avoid TrtAllReduce deadlock.
  std::shared_ptr<Waiter> prev_mtp_done_;

  // Protect concurrent modification of infer_request by MTP and UpdateInferRequest.
  // Non-owning: must be ExecutorRuntime::infer_req_mutex_, the same instance UpdateCachedRequests
  // locks, otherwise the two threads do not actually mutually exclude each other.
  std::mutex* infer_req_mutex_;
};

}  // namespace ksana_llm