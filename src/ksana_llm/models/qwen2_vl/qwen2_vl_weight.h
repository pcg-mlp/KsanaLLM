/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/models/common/common_weight.h"

namespace ksana_llm {

template <typename T>
class Qwen2VLWeight : public CommonWeight<T> {
 public:
  Qwen2VLWeight() {}
  explicit Qwen2VLWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);
};

}  // namespace ksana_llm
