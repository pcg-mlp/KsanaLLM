/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

#include "numerous_llm/utils/environment.h"

namespace numerous_llm {

class ModelInstance {
public:
  // Load model with specified model config.
  void Load(const ModelConfig &model_config);

  // The instance name.
  std::string name;

  // forward
  void Forward(const InferRequest& infer_req);
};

} // namespace numerous_llm
