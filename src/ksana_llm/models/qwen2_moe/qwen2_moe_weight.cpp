/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/qwen2_moe/qwen2_moe_weight.h"

namespace ksana_llm {

template <typename T>
Qwen2MoeWeight<T>::Qwen2MoeWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context) {
  common_moe_weight_ = std::make_shared<CommonMoeWeight<T>>(model_config, rank, context);
}

template <typename T>
Tensor Qwen2MoeWeight<T>::GetModelWeights(const std::string& weight_name) {
  return common_moe_weight_->GetModelWeights(weight_name);
}

template <typename T>
Status Qwen2MoeWeight<T>::LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader,
                                              std::vector<std::string>& weight_name_list,
                                              std::vector<std::string>& custom_name_list) {
  if (!common_moe_weight_->LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list).OK()) {
    KLLM_THROW(fmt::format("Load weight file {} error.", weights_loader->GetTensorFileName()));
  }
  return Status();
}

template <typename T>
void Qwen2MoeWeight<T>::ProcessWeights() {
  common_moe_weight_->ProcessWeights();
}

template <typename T>
void Qwen2MoeWeight<T>::SetEmbeddingsConfig() {
  common_moe_weight_->SetEmbeddingsConfig();
}

template class Qwen2MoeWeight<float>;
template class Qwen2MoeWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class Qwen2MoeWeight<bfloat16>;
#endif

}  // namespace ksana_llm