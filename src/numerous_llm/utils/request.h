/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <sys/stat.h>
#include <vector>

#include "numerous_llm/utils/id_generator.h"
#include "numerous_llm/utils/tensor.h"
#include "numerous_llm/utils/waiter.h"

namespace numerous_llm {

class SamplingConfig {};

class Request {
 public:
  Request();

  // The unique id of a request.
  int64_t req_id;

  // The tokens of this request.
  std::vector<std::vector<int>> tokens;

  // The config of sampling.
  std::vector<SamplingConfig> sampling_configs;

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
  std::vector<std::vector<int>> tokens;
};

}  // namespace numerous_llm
