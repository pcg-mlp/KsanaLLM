/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/common_moe/common_moe_model.h"

namespace ksana_llm {

template <typename T>
CommonMoeModel<T>::CommonMoeModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context)
    : CommonModel<T>(model_config, rank, context) {}

template <typename T>
void CommonMoeModel<T>::InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight) {
  GetBlockManager()->SetDeviceId(rank_);
#ifdef ENABLE_CUDA
  DataType weight_type = model_config_.weight_data_type;
  DataType input_type = weight_type;
  DataType output_type = weight_type;

  moe_layer_ = matmul_layer_factory_->AutoCreateLayer(
      base_weight,
      std::vector<std::string>{"model.layers.0.mlp.experts.up_gate_proj.weight",
                               "model.layers.0.mlp.experts.down_proj.weight"},
      weight_type, input_type, output_type, {model_run_config.moe_scale_norm_mode});

  expert_gating_layer_ = matmul_layer_factory_->AutoCreateLayer(base_weight, "model.layers.0.mlp.gate.weight",
                                                                weight_type, input_type, output_type, {});
#endif

  CommonModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status CommonMoeModel<T>::CommonMlp(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                    const std::vector<Tensor>& mlp_input, const bool is_multi_token_forward) {
#ifdef ENABLE_CUDA
  // Expert share gating MatMul
  Tensor expert_gating_weight = base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.gate.weight", layer_idx));
  STATUS_CHECK_RETURN(expert_gating_layer_->Forward({mlp_input[0], expert_gating_weight}, gated_buffer_));

  // MOE layer
  Tensor expert_up_gate_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.experts.up_gate_proj.weight", layer_idx));
  Tensor expert_down_weight =
      base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.experts.down_proj.weight", layer_idx));
  STATUS_CHECK_RETURN(moe_layer_->Forward({mlp_input[0], gated_buffer_[0], expert_up_gate_weight, expert_down_weight},
                                          hidden_buffer_1_));

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(model_output_->compute_ready_event, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetCommStreams()[rank_], model_output_->compute_ready_event);
  }

  // Mlp AllReduceSum
  if (model_communicator_) {
    model_communicator_->ReduceSum(hidden_buffer_1_, hidden_buffer_0_, is_multi_token_forward, true);
  }
#endif
  return Status();
}

template class CommonMoeModel<float>;
template class CommonMoeModel<float16>;
#ifdef ENABLE_BFLOAT16
template class CommonMoeModel<bfloat16>;
#endif

}  // namespace ksana_llm