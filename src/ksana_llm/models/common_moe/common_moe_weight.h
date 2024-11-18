/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/models/common/common_weight.h"

namespace ksana_llm {

template <typename T>
class CommonMoeWeight : public CommonWeight<T> {
 public:
  CommonMoeWeight() {}
  explicit CommonMoeWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);

  Status LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader,
                             std::vector<std::string>& weight_name_list,
                             std::vector<std::string>& custom_name_list) override;

  void ProcessWeights() override;

 protected:
  using CommonWeight<T>::context_;
  using CommonWeight<T>::rank_;
  using CommonWeight<T>::tensor_para_size_;

  using CommonWeight<T>::moe_weight_data_type_;
  using CommonWeight<T>::weights_map_;
  using CommonWeight<T>::weights_data_type_map_;

  using CommonWeight<T>::model_config_;

  using CommonWeight<T>::tensor_manager_;

 private:
  Status GetExpertsIdx(const std::string& expert_name);

  Status PermuteGatingWeight(Tensor& last_gating_tensor, const int num_layer, const bool is_share_gating);

  Status PermuteShareMLPWeight(Tensor& last_share_down_up_tensor, Tensor& last_share_gate_tensor, const int num_layer);

  int layer_idx_;
  int expert_idx_;
};

}  // namespace ksana_llm
