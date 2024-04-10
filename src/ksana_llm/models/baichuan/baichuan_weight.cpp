/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/baichuan/baichuan_weight.h"

namespace ksana_llm {

template <typename T>
BaichuanWeight<T>::BaichuanWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context) {
  common_weight_ = std::make_shared<CommonWeight<T>>(model_config, rank, context);
}

template <typename T>
Tensor BaichuanWeight<T>::GetModelWeights(const std::string& weight_name) {
  return common_weight_->GetModelWeights(weight_name);
}

template class BaichuanWeight<float>;
template class BaichuanWeight<float16>;

}  // namespace ksana_llm
