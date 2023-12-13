/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/utils/environment.h"

namespace numerous_llm {

class BaseWeight {
 public:
  BaseWeight(){};
  explicit BaseWeight(const ModelConfig& model_config);
  ~BaseWeight(){};
};

}  // namespace numerous_llm
