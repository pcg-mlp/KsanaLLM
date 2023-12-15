/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <sys/stat.h>
#include <vector>

#include "numerous_llm/utils/id_generator.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

class SamplingConfig {};

class Request {
 public:
  Request();

  // The unique id of a request.
  int64_t req_id;

  // The tokens of this request.
  std::vector<std::vector<int>> tokens;

  // The tensors of this request, on cpu.
  std::vector<TensorMap> tensor_maps;

  // The config of sampling.
  std::vector<SamplingConfig> sampling_configs;

 private:
  // The id generator
  static IdGenerator id_generator_;
};

class Response {
 public:
  // The unique id of a request.
  int64_t req_id;

  // The tensors of this request, on cpu.
  std::vector<TensorMap> tensor_maps;
};

}  // namespace numerous_llm
