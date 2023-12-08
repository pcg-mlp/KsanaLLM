/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <string>

#include "numerous_llm/runtime/model_instance.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/request.h"

#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

// The infer request, it is the unit of batch manager's scheduler.
class InferRequest {
 public:
  InferRequest();

  // The req id of the user's request.
  int req_id;

  // The unique id of this request.
  int infer_id;

  // The name of model instance.
  std::string model_name;

  // The model instance pointer.
  std::shared_ptr<ModelInstance> model_instance;

  // The arrive time
  unsigned long timestamp_in_ms;

  // The input tensor map.
  TensorMap input_tensor_map;

  // The sampling config of this request.
  SamplingConfig sampling_config;

  // The output tensor map.
  TensorMap output_tensor_map;
};

}  // namespace numerous_llm
