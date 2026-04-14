/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/common_moe/common_moe_weight.h"

namespace ksana_llm {

template <typename T>
class Glm4MoeWeight : public CommonMoeWeight<T> {
 public:
  Glm4MoeWeight() {}
  explicit Glm4MoeWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                         std::shared_ptr<Context> context);

  Status LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                             const std::vector<std::string>& weight_name_list,
                             const std::vector<std::string>& custom_name_list) override;

  void ProcessWeights() override;

 protected:
  using BaseWeight::context_;
  using BaseWeight::model_config_;
  using BaseWeight::rank_;
  using BaseWeight::weights_data_type_map_;
  using BaseWeight::weights_map_;
  using CommonWeight<T>::required_layer_idx_;
  using CommonWeight<T>::quant_weight_solver_;
  using CommonWeight<T>::tensor_manager_;
  using CommonWeight<T>::weight_data_type_;
  using CommonWeight<T>::CommonPermuteWeight;
};

}  // namespace ksana_llm
