/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common_moe/common_moe_model.h"

namespace ksana_llm {

template <typename T>
class __attribute__((visibility("hidden"))) MixtralModel : public BaseModel {
 public:
  MixtralModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
               std::shared_ptr<BaseWeight> base_weight);

  float* GetLogitsPtr();

  // Forward.
  Status Forward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs);

 private:
  // The commonmoe model instance.
  std::shared_ptr<CommonMoeModel<T>> common_moe_model_ = nullptr;
};

}  // namespace ksana_llm
