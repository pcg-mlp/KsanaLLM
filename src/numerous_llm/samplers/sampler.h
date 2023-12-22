/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/runtime/sampling_request.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class Sampler {
 public:
  Sampler(int rank) : rank_(rank) {}

  Status Sampling(std::vector<SamplingRequest>& sampling_reqs);

 private:
  int rank_;
};

}  // namespace numerous_llm
