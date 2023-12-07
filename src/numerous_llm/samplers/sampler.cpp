/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/samplers/sampler.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

Status Sampler::Sampling(std::vector<InferRequest> &reqs) {
  NLLM_LOG_INFO << "llm sampler invoked." << std::endl;
  return Status();
}

}  // namespace numerous_llm
