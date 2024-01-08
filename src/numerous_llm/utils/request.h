/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <sys/stat.h>
#include <vector>

#include "numerous_llm/utils/id_generator.h"
#include "numerous_llm/utils/tensor.h"
#include "numerous_llm/utils/waiter.h"

namespace numerous_llm {

struct SamplingConfig {
  int beam_width;
  int topk;
  float topp;
  float temperature;
};

class Request {
 public:
  Request();

  // The unique id of a request.
  int64_t req_id;

  // The requested model name.
  std::string model_name;

  // The tokens of this request.
  std::vector<int> input_tokens;

  // The config of sampling.
  SamplingConfig sampling_config;

  std::shared_ptr<Waiter> waiter;

 private:
  // The id generator
  static IdGenerator id_generator_;
};

class Response {
 public:
  // The unique id of a request, same as its request.
  int64_t req_id;

  // The tokens of this response.
  std::vector<int> output_tokens;
};

}  // namespace numerous_llm
