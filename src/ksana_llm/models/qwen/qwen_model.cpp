/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/qwen/qwen_model.h"

namespace ksana_llm {

template <typename T>
QwenModel<T>::QwenModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
                        std::shared_ptr<BaseWeight> base_weight) {
  common_model_ = std::make_shared<CommonModel<T>>(model_config, rank, context);

  ModelRunConfig model_run_config;
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.qkv_add_bias = true;
  common_model_->InitRunConfig(model_run_config, base_weight);
}

template <typename T>
float* QwenModel<T>::GetLogitsPtr() {
  return common_model_->GetLogitsPtr();
}

template <typename T>
Status QwenModel<T>::Forward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                             std::vector<ForwardRequest>& forward_reqs, bool epilogue) {
  return common_model_->Forward(base_weight, forward_reqs, epilogue);
}

template class QwenModel<float>;
template class QwenModel<float16>;
#ifdef ENABLE_BFLOAT16
template class QwenModel<bfloat16>;
#endif

}  // namespace ksana_llm
