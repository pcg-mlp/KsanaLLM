/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/llama/llama_model.h"

namespace ksana_llm {

template <typename T>
LlamaModel<T>::LlamaModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context) {
  common_model_ = std::make_shared<CommonModel<T>>(model_config, rank, context);

  ModelRunConfig model_run_config;
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.qkv_add_bias = false;
  common_model_->InitRunConfig(model_run_config);
}

template <typename T>
float* LlamaModel<T>::GetLogitsPtr() {
  return common_model_->GetLogitsPtr();
}

template <typename T>
Status LlamaModel<T>::ContextDecode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                    std::vector<ForwardRequest>& forward_reqs) {
  return common_model_->ContextDecode(base_weight, forward_reqs);
}

template <typename T>
Status LlamaModel<T>::Decode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                             std::vector<ForwardRequest>& forward_reqs) {
  return common_model_->Decode(base_weight, forward_reqs);
}

template class LlamaModel<float>;
template class LlamaModel<float16>;
#ifdef ENABLE_BFLOAT16
template class LlamaModel<bfloat16>;
#endif

}  // namespace ksana_llm
