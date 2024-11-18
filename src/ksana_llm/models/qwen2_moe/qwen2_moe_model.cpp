/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/qwen2_moe/qwen2_moe_model.h"

namespace ksana_llm {

template <typename T>
Qwen2MoeModel<T>::Qwen2MoeModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
                                std::shared_ptr<BaseWeight> base_weight)
    : CommonMoeModel<T>(model_config, rank, context) {
  ModelRunConfig model_run_config;
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.moe_scale_norm_mode = MoeScaleNormMode::NO_NORM;
  model_run_config.qkv_add_bias = true;
  InitRunConfig(model_run_config, base_weight);
}

template <typename T>
void Qwen2MoeModel<T>::InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight) {
  GetBlockManager()->SetDeviceId(rank_);
#ifdef ENABLE_CUDA
  if (model_config_.has_shared_experts) {
    DataType weight_type = model_config_.weight_data_type;
    DataType input_type = weight_type;
    DataType output_type = weight_type;

    size_t max_token_num = model_config_.max_scheduler_token_num;
    moe_buffer_size_ = max_token_num * model_config_.hidden_units;
    STATUS_CHECK_FAILURE(CommonModel<T>::CreateBufferTensor(moe_buffer_[0], {moe_buffer_size_}, weight_type));
    STATUS_CHECK_FAILURE(CommonModel<T>::CreateBufferTensor(share_gating_buffer_[0], {max_token_num}, weight_type));

    share_expert_gating_layer_ = matmul_layer_factory_->AutoCreateLayer(
        base_weight, "model.layers.0.mlp.shared_expert_gate.weight", weight_type, input_type, output_type, {});

    share_expert_gate_proj_layer_ = matmul_layer_factory_->AutoCreateLayer(
        base_weight, "model.layers.0.mlp.shared_expert.gate_proj.weight", weight_type, input_type, output_type, {});

    share_expert_up_proj_layer_ = matmul_layer_factory_->AutoCreateLayer(
        base_weight, "model.layers.0.mlp.shared_expert.up_proj.weight", weight_type, input_type, output_type, {});

    share_expert_down_proj_layer_ = matmul_layer_factory_->AutoCreateLayer(
        base_weight, "model.layers.0.mlp.shared_expert.down_proj.weight", weight_type, input_type, output_type, {});

    sigmoid_layer_ = std::make_shared<SigmoidLayer<T>>();
    sigmoid_layer_->Init({}, context_, rank_);
    mul_layer_ = std::make_shared<MulLayer<T>>();
    mul_layer_->Init({}, context_, rank_);
  }
#endif

  CommonMoeModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status Qwen2MoeModel<T>::CommonMlp(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                   const std::vector<Tensor>& mlp_input, const bool is_context_stage) {
  if (model_config_.has_shared_experts) {
#ifdef ENABLE_CUDA
    // Expert gating MatMul
    Tensor expert_gating_weight =
        base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.gate.weight", layer_idx));
    STATUS_CHECK_RETURN(expert_gating_layer_->Forward({mlp_input[0], expert_gating_weight}, gated_buffer_));

    // MOE layer
    Tensor expert_up_gate_weight =
        base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.experts.up_gate_proj.weight", layer_idx));
    Tensor expert_down_weight =
        base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.experts.down_proj.weight", layer_idx));
    STATUS_CHECK_RETURN(
        moe_layer_->Forward({mlp_input[0], gated_buffer_[0], expert_up_gate_weight, expert_down_weight}, moe_buffer_));

    // Expert share gating MatMul
    Tensor share_expert_gating_weight =
        base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.shared_expert_gate.weight", layer_idx));
    STATUS_CHECK_RETURN(
        share_expert_gating_layer_->Forward({mlp_input[0], share_expert_gating_weight}, share_gating_buffer_));

    // Expert gate_proj MatMul
    Tensor share_expert_gate_proj_weight =
        base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.shared_expert.gate_proj.weight", layer_idx));
    STATUS_CHECK_RETURN(
        share_expert_gate_proj_layer_->Forward({mlp_input[0], share_expert_gate_proj_weight}, hidden_buffer_1_));
    // Expert up_proj MatMul
    Tensor share_expert_up_proj_weight =
        base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.shared_expert.up_proj.weight", layer_idx));
    STATUS_CHECK_RETURN(
        share_expert_up_proj_layer_->Forward({mlp_input[0], share_expert_up_proj_weight}, gated_buffer_));
    std::swap(hidden_buffer_1_, hidden_buffer_0_);

    // Activation is an in-place operation, just put the output in `hidden_buffer_0_`, the same as the input.
    STATUS_CHECK_RETURN(silu_mul_layer_->Forward({hidden_buffer_0_[0], gated_buffer_[0]}, hidden_buffer_0_));

    // Expert down_proj MatMul
    Tensor share_expert_down_proj_weight =
        base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.shared_expert.down_proj.weight", layer_idx));
    if (model_communicator_) {
      // Put output to `reduce_buffer_` to ensure that the input for custom reduce sum is always in `reduce_buffer_`.
      STATUS_CHECK_RETURN(
          share_expert_down_proj_layer_->Forward({hidden_buffer_0_[0], share_expert_down_proj_weight}, reduce_buffer_));
    } else {
      STATUS_CHECK_RETURN(share_expert_down_proj_layer_->Forward({hidden_buffer_0_[0], share_expert_down_proj_weight},
                                                                 hidden_buffer_1_));
      std::swap(hidden_buffer_1_, hidden_buffer_0_);
    }

    // Expert share gating sigmoid
    STATUS_CHECK_RETURN(sigmoid_layer_->Forward(share_gating_buffer_, share_gating_buffer_));
    // Expert share gating mul
    if (model_communicator_) {
      STATUS_CHECK_RETURN(mul_layer_->Forward({share_gating_buffer_[0], reduce_buffer_[0]}, hidden_buffer_1_));
      STATUS_CHECK_RETURN(add_layer_->Forward({hidden_buffer_1_[0], moe_buffer_[0]}, reduce_buffer_));
    } else {
      STATUS_CHECK_RETURN(mul_layer_->Forward({share_gating_buffer_[0], hidden_buffer_0_[0]}, hidden_buffer_1_));
      STATUS_CHECK_RETURN(add_layer_->Forward({hidden_buffer_1_[0], moe_buffer_[0]}, hidden_buffer_0_));
    }

    // NOTE(karlluo): multiple event in nccl will cause preformance regression
    // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
    if (!context_->IsRunContextDecodeAndDecodeSerially()) {
      EventRecord(model_output_->compute_ready_event, context_->GetComputeStreams()[rank_]);
      StreamWaitEvent(context_->GetNCCLStreams()[rank_], model_output_->compute_ready_event);
    }

    // Mlp AllReduceSum
    if (model_communicator_) {
      model_communicator_->ReduceSum(reduce_buffer_, hidden_buffer_0_, is_context_stage, true);
    }
#endif
  } else {
    CommonMoeModel<T>::CommonMlp(layer_idx, base_weight, mlp_input, is_context_stage);
  }
  return Status();
}

template class Qwen2MoeModel<float>;
template class Qwen2MoeModel<float16>;
#ifdef ENABLE_BFLOAT16
template class Qwen2MoeModel<bfloat16>;
#endif
}  // namespace ksana_llm