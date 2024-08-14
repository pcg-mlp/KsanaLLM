/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <vector>

#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/utils/request.h"

namespace ksana_llm {

// The information used for sampling.
struct SamplingRequest {
  // The req id of the user's request.
  int64_t req_id;

  // The custom length for the logits output, allowing for a specific size of logits to be generated.
  size_t logits_custom_length = 0;

  // The sampling config.
  SamplingConfig* sampling_config;

  // The logits buf and offset.
  std::vector<float*> logits_buf;
  size_t logits_offset;

  std::vector<int>* input_tokens;
  // The output token will be appended here.
  std::vector<int>* output_tokens;

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  const std::map<std::string, TargetDescribe>* request_target = nullptr;

  // The result of request_target.
  std::map<std::string, PythonTensor>* response;

  // The mutex used to protect output_tokens.
  std::mutex* output_mutex;

  // Store token and their corresponding float probability values.
  std::vector<std::vector<std::pair<int, float>>>* logprobs;

  // Beam Search Group
  std::vector<std::shared_ptr<InferRequest>>* req_group;

  // Model config
  const ModelConfig* model_config;
};

}  // namespace ksana_llm
