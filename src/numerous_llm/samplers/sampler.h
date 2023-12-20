/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/runtime/sampling_request.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class Sampler {
 public:
  Status Sampling(std::vector<SamplingRequest> &sampling_reqs);
};

}  // namespace numerous_llm
