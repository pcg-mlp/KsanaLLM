/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "numerous_llm/runtime/context.h"
#include "numerous_llm/runtime/forward_request.h"
#include "numerous_llm/runtime/infer_request.h"
#include "numerous_llm/runtime/worker.h"
#include "numerous_llm/samplers/sampler.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class LlmRuntime {
 public:
  LlmRuntime(std::shared_ptr<Context> contex);

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

 private:
  // The runtime context.
  std::shared_ptr<Context> context_ = nullptr;

  // The worker group for this runtime, do we need several worker_group?
  std::shared_ptr<WorkerGroup> worker_group_ = nullptr;

  // The sampler instance.
  std::shared_ptr<Sampler> sampler_ = nullptr;
};

}  // namespace numerous_llm
