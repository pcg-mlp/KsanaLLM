/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

#include "numerous_llm/models/base/base_model.h"
#include "numerous_llm/models/llama/llama.h"
#include "numerous_llm/runtime/context.h"
#include "numerous_llm/runtime/worker.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

class ModelInstance {
 public:
  ModelInstance(const std::shared_ptr<Context>& context);

  // Load model with specified model config.
  void Load(const ModelConfig& model_config);

  // The instance name.
  std::string name;

  // forward
  void Forward(const TensorMap& input_tensor_map, const SamplingConfig& sampling_config, TensorMap& output_tensor_map);

 private:
  std::shared_ptr<Context> context_{nullptr};
};

}  // namespace numerous_llm
