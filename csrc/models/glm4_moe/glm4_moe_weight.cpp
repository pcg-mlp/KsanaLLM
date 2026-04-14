/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/glm4_moe/glm4_moe_weight.h"

namespace ksana_llm {

template <typename T>
Glm4MoeWeight<T>::Glm4MoeWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                                std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, runtime_config, rank, context),
      CommonMoeWeight<T>(model_config, runtime_config, rank, context) {}

template <typename T>
Status Glm4MoeWeight<T>::LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                             const std::vector<std::string>& weight_name_list,
                                             const std::vector<std::string>& custom_name_list) {
  return CommonMoeWeight<T>::LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
}

template <typename T>
void Glm4MoeWeight<T>::ProcessWeights() {
  // GLM4-MoE uses the new weight loader (Glm4MoeWeightLoader)
  // which handles expert weights directly, so we just call parent's ProcessWeights
  CommonMoeWeight<T>::ProcessWeights();
  CommonWeight<T>::ProcessWeights();
}

template class Glm4MoeWeight<float>;
template class Glm4MoeWeight<float16>;
template class Glm4MoeWeight<bfloat16>;

}  // namespace ksana_llm
