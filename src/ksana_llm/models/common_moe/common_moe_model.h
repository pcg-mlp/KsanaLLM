/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/moe_layer.h"
#include "ksana_llm/models/common/common_model.h"
#include "ksana_llm/models/common/common_weight.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

template <typename T>
class __attribute__((visibility("hidden"))) CommonMoeModel : public CommonModel<T> {
 public:
  CommonMoeModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context);

  // Initialize the run config.
  void InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight);

 protected:
  Status CommonMlp(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                   const std::vector<Tensor>& mlp_input, const bool is_context_stage) override;

 protected:
  using CommonModel<T>::context_;
  using CommonModel<T>::rank_;

  using CommonModel<T>::model_config_;
  using CommonModel<T>::model_output_;
  using CommonModel<T>::model_communicator_;

  using CommonModel<T>::matmul_layer_factory_;

  using CommonModel<T>::hidden_buffer_0_;
  using CommonModel<T>::hidden_buffer_1_;
  using CommonModel<T>::gated_buffer_;

  std::shared_ptr<BaseLayer> moe_layer_;
  std::shared_ptr<BaseLayer> expert_gating_layer_;
};

}  // namespace ksana_llm
