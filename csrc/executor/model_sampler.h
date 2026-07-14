/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "runtime/grammar_output.h"
#include "runtime/infer_request.h"
#include "runtime/model_output.h"
#include "runtime/sampling_output.h"
#include "runtime/sampling_request.h"
#include "samplers/sampler.h"
#include "utils/status.h"

namespace ksana_llm {

class ModelSampler {
 public:
  ModelSampler(std::shared_ptr<Environment> env, std::shared_ptr<Context> context);

  Status Sampling(std::shared_ptr<NewModelOutput> model_output, std::shared_ptr<SamplingOutput> sampling_output);

 private:
  // Build sampling request.
  void BuildSamplingRequest(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs,
                            std::vector<SamplingRequest>& sampling_reqs, const bool enable_main_layers_sampler = true);

  // Generate and apply bitmask from Executor-side generators
  Status GenerateAndApplyBitmask(std::vector<std::shared_ptr<ExecutorInferRequest>>& reqs,
                                 std::vector<SamplingRequest>& sampling_reqs);

 private:
  std::shared_ptr<Environment> env_ = nullptr;
  std::shared_ptr<Context> context_ = nullptr;

  // The sampler instance on every device.
  std::shared_ptr<Sampler> sampler_ = nullptr;

  // vocab size for bitmask generation
  size_t vocab_size_ = 0;
  size_t bitmask_stride_ = 0;
};

}  // namespace ksana_llm
