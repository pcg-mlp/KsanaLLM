/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/worker.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

class LlmRuntime {
 public:
  LlmRuntime(const BatchSchedulerConfig &batch_scheduler_config, std::shared_ptr<Context> context);

  // Execute one req in parallel.
  Status Step(std::vector<std::shared_ptr<InferRequest>> &reqs);

 private:
  // Execute the forward.
  Status Forward(std::vector<std::shared_ptr<InferRequest>> &reqs);

  // Execute the sampling.
  Status Sampling(std::vector<std::shared_ptr<InferRequest>> &reqs);

  // Build forward request, group by model name and stage.
  void BuildForwardRequests(
      std::vector<std::shared_ptr<InferRequest>> &reqs,
      std::unordered_map<ModelInstance *, std::unordered_map<InferStage, std::vector<ForwardRequest>>> &grouped_reqs);

  // Build sampling request.
  void BuildSamplingRequest(std::vector<std::shared_ptr<InferRequest>> &reqs,
                            std::vector<SamplingRequest> &sampling_reqs);

  // Run context decode and decode serially in single thread.
  Status RunContextDecodeAndDecodeRunSerially(
      std::unordered_map<ModelInstance *, std::unordered_map<InferStage, std::vector<ForwardRequest>>> &grouped_reqs);

 private:
  BatchSchedulerConfig batch_schedule_config_;

  // The runtime context.
  std::shared_ptr<Context> context_ = nullptr;

  // The worker group for this runtime, do we need several worker_group?
  std::shared_ptr<WorkerGroup> worker_group_ = nullptr;

  // The sampler instance on every device.
  std::vector<std::shared_ptr<Sampler>> samplers_;
};

}  // namespace ksana_llm
