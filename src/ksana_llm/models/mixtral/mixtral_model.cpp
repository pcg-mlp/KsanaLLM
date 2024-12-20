/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/mixtral/mixtral_model.h"

namespace ksana_llm {

template <typename T>
MixtralModel<T>::MixtralModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
                              std::shared_ptr<BaseWeight> base_weight) {
  common_moe_model_ = std::make_shared<CommonMoeModel<T>>(model_config, rank, context);

  ModelRunConfig model_run_config;
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.moe_scale_norm_mode = MoeScaleNormMode::RE_NORM;
  model_run_config.qkv_add_bias = false;
  common_moe_model_->InitRunConfig(model_run_config, base_weight);
}

template <typename T>
float* MixtralModel<T>::GetLogitsPtr() {
  return common_moe_model_->GetLogitsPtr();
}

template <typename T>
Status MixtralModel<T>::Forward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                std::vector<ForwardRequest>& forward_reqs, bool epilogue) {
  return common_moe_model_->Forward(base_weight, forward_reqs, epilogue);
}

template class MixtralModel<float>;
template class MixtralModel<float16>;
#ifdef ENABLE_BFLOAT16
template class MixtralModel<bfloat16>;
#endif

}  // namespace ksana_llm
