/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/models/common_moe/common_moe_weight.h"

namespace ksana_llm {

template <typename T>
class Qwen2MoeWeight : public BaseWeight {
 public:
  Qwen2MoeWeight() {}
  explicit Qwen2MoeWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);

  Tensor GetModelWeights(const std::string& weight_name);

  Status LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader,
                             std::vector<std::string>& weight_name_list, std::vector<std::string>& custom_name_list);

  void ProcessWeights();

  void SetEmbeddingsConfig();

 private:
  // the moe weight instance.
  std::shared_ptr<CommonMoeWeight<T>> common_moe_weight_ = nullptr;
};

}  // namespace ksana_llm
