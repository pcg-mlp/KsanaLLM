/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

class SamplingConfig {};

class Request {
 public:
  // The unique id of a request.
  int req_id;

  // The tensors of this request, on cpu.
  std::vector<TensorMap> tensor_maps;

  // The config of sampling.
  std::vector<SamplingConfig> sampling_configs;
};

class Response {
 public:
  // The unique id of a request.
  int req_id;

  // The tensors of this request, on cpu.
  std::vector<TensorMap> tensor_maps;
};

}  // namespace numerous_llm
