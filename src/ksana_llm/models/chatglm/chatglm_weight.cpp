/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/chatglm/chatglm_weight.h"

namespace ksana_llm {

template <typename T>
ChatglmWeight<T>::ChatglmWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context) {
  common_weight_ = std::make_shared<CommonWeight<T>>(model_config, rank, context);
}

template <typename T>
Tensor ChatglmWeight<T>::GetModelWeights(const std::string& weight_name) {
  return common_weight_->GetModelWeights(weight_name);
}

template <typename T>
Status ChatglmWeight<T>::LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader,
                                              std::vector<std::string>& weight_name_list,
                                              std::vector<std::string>& custom_name_list) {
  if (!common_weight_->LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list).OK()) {
    NLLM_LOG_ERROR << fmt::format("Load weight file error.");
    exit(-1);
  }
  return Status();
}

template <typename T>
void ChatglmWeight<T>::ProcessWeights() {
  common_weight_->ProcessWeights();
}

template class ChatglmWeight<float>;
template class ChatglmWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class ChatglmWeight<bfloat16>;
#endif

}  // namespace ksana_llm

