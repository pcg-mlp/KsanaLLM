/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/models/base/base_weight.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/utils.h"

namespace numerous_llm {

class LlamaWeight : public BaseWeight {
 public:
  explicit LlamaWeight(const ModelConfig& model_config);
  ~LlamaWeight();
};

}  // namespace numerous_llm
