/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

class BaseWeight {
 public:
  BaseWeight(){};
  explicit BaseWeight(const ModelConfig& model_config, int rank);
  ~BaseWeight(){};

  // 查表,返回 weights_map_[weight_name]
  virtual Tensor GetModelWeights(std::string& weight_name) = 0;
};

}  // namespace numerous_llm
