/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/samplers/sampler.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

Status Sampler::Sampling(std::vector<SamplingRequest> &sampling_reqs) {
  NLLM_LOG_INFO << "llm sampler invoked.";

  for (auto &req : sampling_reqs) {
    // TODO(karlluo): just a fake result for scheduler output result
    req.output_tokens->push_back(1);
  }

  return Status();
}

}  // namespace numerous_llm
