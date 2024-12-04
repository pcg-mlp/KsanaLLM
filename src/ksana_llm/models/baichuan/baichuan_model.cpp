/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/baichuan/baichuan_model.h"
#include "ksana_llm/models/common/common_model.h"

namespace ksana_llm {

template <typename T>
BaichuanModel<T>::BaichuanModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
                                std::shared_ptr<BaseWeight> base_weight) {
  common_model_ = std::make_shared<CommonModel<T>>(model_config, rank, context);

  ModelRunConfig model_run_config;
  // The Baichuan1-7B and Baichuan2-7B models do not use the Alibi mode for loading,
  // and both of these models have a hidden_units value of 4096.
  model_run_config.position_encoding =
      (model_config.size_per_head * model_config.head_num != 4096) ? PositionEncoding::ALIBI : PositionEncoding::ROPE;
  model_run_config.qkv_add_bias = false;
  common_model_->InitRunConfig(model_run_config, base_weight);
}

template <typename T>
float* BaichuanModel<T>::GetLogitsPtr() {
  return common_model_->GetLogitsPtr();
}

template <typename T>
Status BaichuanModel<T>::Forward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                 std::vector<ForwardRequest>& forward_reqs) {
  return common_model_->Forward(base_weight, forward_reqs);
}

template class BaichuanModel<float>;
template class BaichuanModel<float16>;
#ifdef ENABLE_BFLOAT16
template class BaichuanModel<bfloat16>;
#endif

}  // namespace ksana_llm
