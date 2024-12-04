/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/activation_layer.h"
#include "ksana_llm/layers/mul_layer.h"
#include "ksana_llm/models/common_moe/common_moe_model.h"
#include "ksana_llm/models/common_moe/common_moe_weight.h"

namespace ksana_llm {

template <typename T>
class __attribute__((visibility("hidden"))) Qwen2MoeModel : public CommonMoeModel<T> {
 public:
  Qwen2MoeModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
                std::shared_ptr<BaseWeight> base_weight);

  // Initialize the run config.
  void InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight);

 protected:
  Status CommonMlp(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                   const std::vector<Tensor>& mlp_input, const bool is_multi_token_forward) override;

 protected:
  using CommonModel<T>::context_;
  using CommonModel<T>::rank_;

  using CommonModel<T>::model_config_;
  using CommonModel<T>::model_output_;
  using CommonModel<T>::model_communicator_;

  using CommonModel<T>::matmul_layer_factory_;
  using CommonModel<T>::add_layer_;
  using CommonModel<T>::silu_mul_layer_;

  using CommonModel<T>::hidden_buffer_0_;
  using CommonModel<T>::hidden_buffer_1_;
  using CommonModel<T>::reduce_buffer_;
  using CommonModel<T>::gated_buffer_;

  using CommonMoeModel<T>::moe_layer_;
  using CommonMoeModel<T>::expert_gating_layer_;

 private:
  size_t moe_buffer_size_;

  std::vector<Tensor> moe_buffer_{1};
  std::vector<Tensor> share_gating_buffer_{1};

  std::shared_ptr<BaseLayer> share_expert_gating_layer_;
  std::shared_ptr<BaseLayer> share_expert_gate_proj_layer_;
  std::shared_ptr<BaseLayer> share_expert_up_proj_layer_;
  std::shared_ptr<BaseLayer> share_expert_down_proj_layer_;

  std::shared_ptr<SigmoidLayer<T>> sigmoid_layer_;
  std::shared_ptr<BaseLayer> mul_layer_;
};

}  // namespace ksana_llm
