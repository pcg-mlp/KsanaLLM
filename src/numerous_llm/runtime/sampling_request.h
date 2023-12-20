/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <vector>

#include "numerous_llm/utils/request.h"

namespace numerous_llm {

// The information used for sampling.
struct SamplingRequest {
  // The sampling config.
  SamplingConfig* sampling_config;

  // The logitst buf and offset.
  std::vector<float*> logits_buf;
  size_t logits_offset;

  // The output token will be appended here.
  std::vector<int>* output_tokens;
};

}  // namespace numerous_llm
