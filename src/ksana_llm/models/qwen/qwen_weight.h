/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/models/common/common_weight.h"

namespace ksana_llm {

template <typename T>
class QwenWeight : public BaseWeight {
 public:
  QwenWeight() {}
  explicit QwenWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);

  Tensor GetModelWeights(const std::string& weight_name);

 private:
  // the common weight instance.
  std::shared_ptr<CommonWeight<T>> common_weight_ = nullptr;
};

}  // namespace ksana_llm
