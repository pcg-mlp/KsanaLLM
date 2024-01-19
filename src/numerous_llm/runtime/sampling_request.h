/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <vector>

#include "numerous_llm/utils/request.h"

namespace numerous_llm {

// The information used for sampling.
struct SamplingRequest {
  // The req id of the user's request.
  int64_t req_id;

  // The sampling config.
  SamplingConfig* sampling_config;

  // The logitst buf and offset.
  std::vector<float*> logits_buf;
  size_t logits_offset;

  // The output token will be appended here.
  std::vector<int>* output_tokens;

  // The mutex used to protect output_tokens.
  std::mutex* output_mutex;

  // Model config
  const ModelConfig* model_config;
};

}  // namespace numerous_llm
