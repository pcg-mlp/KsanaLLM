/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/baichuan/baichuan_model.h"
#include "ksana_llm/models/common/common_model.h"

namespace ksana_llm {

template <typename T>
BaichuanModel<T>::BaichuanModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context) {
  common_model_ = std::make_shared<CommonModel<T>>(model_config, rank, context);

  ModelRunConfig model_run_config;
  // The Baichuan1-7B and Baichuan2-7B models do not use the Alibi mode for loading,
  // and both of these models have a hidden_units value of 4096.
  model_run_config.position_encoding =
      (model_config.size_per_head * model_config.head_num != 4096) ? PositionEncoding::ALIBI : PositionEncoding::ROPE;
  model_run_config.qkv_add_bias = false;
  common_model_->InitRunConfig(model_run_config);
}

template <typename T>
float* BaichuanModel<T>::GetLogitsPtr() {
  return common_model_->GetLogitsPtr();
}

template <typename T>
Status BaichuanModel<T>::ContextDecode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                       std::vector<ForwardRequest>& forward_reqs) {
  return common_model_->ContextDecode(base_weight, forward_reqs);
}

template <typename T>
Status BaichuanModel<T>::Decode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                std::vector<ForwardRequest>& forward_reqs) {
  return common_model_->Decode(base_weight, forward_reqs);
}

template class BaichuanModel<float>;
template class BaichuanModel<float16>;

}  // namespace ksana_llm
