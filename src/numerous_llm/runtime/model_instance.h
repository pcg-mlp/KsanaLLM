/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

class ModelInstance {
 public:
  // Load model with specified model config.
  void Load(const ModelConfig& model_config);

  // The instance name.
  std::string name;

  // forward
  void Forward(const TensorMap& input_tensor_map, const SamplingConfig& sampling_config, TensorMap& output_tensor_map);
};

}  // namespace numerous_llm
