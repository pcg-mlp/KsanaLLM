/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/utils/environment.h"

namespace numerous_llm {

class BaseWeight {
 public:
  // Disable a default constructor
  BaseWeight();
  explicit BaseWeight(const ModelConfig& model_config);

  virtual ~BaseWeight() = 0;
};

}  // namespace numerous_llm
