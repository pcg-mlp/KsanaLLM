/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/llama/llama_weight.h"

namespace ksana_llm {

template <typename T>
LlamaWeight<T>::LlamaWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context) {
  common_weight_ = std::make_shared<CommonWeight<T>>(model_config, rank, context);
}

template <typename T>
Tensor LlamaWeight<T>::GetModelWeights(const std::string& weight_name) {
  return common_weight_->GetModelWeights(weight_name);
}

template class LlamaWeight<float>;
template class LlamaWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class LlamaWeight<bfloat16>;
#endif

}  // namespace ksana_llm
