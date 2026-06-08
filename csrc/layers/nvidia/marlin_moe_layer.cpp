/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/
#include "layers/marlin_moe_layer.h"
#include "kernels/nvidia/kernel_wrapper.h"
#include "utils/utils.h"

namespace ksana_llm {

Status MarlinMoeLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  inter_data_type_ = runtime_config.inter_data_type;

  int parameter_index = 0;
  moe_scale_norm_mode_ = std::any_cast<const MoeScaleNormMode>(parameters[parameter_index++]);
  max_token_num_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_num_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_hidden_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_inter_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_topk_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  tp_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  use_vllm_moe_ = std::any_cast<bool>(parameters[parameter_index++]);
  num_expert_group_ = std::any_cast<uint32_t>(parameters[parameter_index++]);
  expert_groups_topk_ = std::any_cast<uint32_t>(parameters[parameter_index++]);
  scoring_func_ = std::any_cast<std::string>(parameters[parameter_index++]);
  topk_method_ = std::any_cast<std::string>(parameters[parameter_index++]);
  norm_topk_prob_ = std::any_cast<bool>(parameters[parameter_index++]);
  routed_scaling_factor_ = std::any_cast<float>(parameters[parameter_index++]);
  use_e_score_correction_bias_ = std::any_cast<bool>(parameters[parameter_index++]);
  enable_full_shared_expert_ = std::any_cast<bool>(parameters[parameter_index++]);
  DataType fp8_weight_dtype = std::any_cast<DataType>(parameters[parameter_index++]);
  DataType int_weight_dtype = std::any_cast<DataType>(parameters[parameter_index++]);
  group_size_ = std::any_cast<int>(parameters[parameter_index++]);
  apply_weight_ = std::any_cast<bool>(parameters[parameter_index++]);
  const float swiglu_limit = std::any_cast<float>(parameters[parameter_index++]);
  KLLM_CHECK_WITH_INFO(!(swiglu_limit > 0.0f), "MarlinMoeLayer does not support swiglu_limit.");
  return Status();
}

size_t MarlinMoeLayer::GetWorkspaceSize() { DISPATCH_BY_3_DTYPE(inter_data_type_, GetWorkspaceSizeT); }

template <typename T>
size_t MarlinMoeLayer::GetWorkspaceSizeT() {
  max_gating_size_ = sizeof(float) * max_token_num_ * expert_num_;
  max_ws_bytes_ = max_gating_size_;
  max_ws_bytes_ += InvokeGetFusedMarlinMoeWorkspaceSize(max_token_num_, expert_inter_size_, expert_hidden_size_,
                                                        expert_num_, expert_topk_, sizeof(T));
  return max_ws_bytes_;
}

Status MarlinMoeLayer::Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) {
  return Status();
}

Status MarlinMoeLayer::Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1,
                               const Tensor* w2, const Tensor* /*eplb_map*/, Tensor* output, Tensor* /*workspace*/,
                               const Tensor* /*e_score_bias*/, const Tensor* /*input_ids*/,
                               const LayerExecContext& exec_ctx) {
  if (hidden_states->dtype != TYPE_FP16) {
    KLLM_THROW("marlin moe only support input type " + std::to_string(TYPE_FP16) + ", not support input type " +
               std::to_string(hidden_states->dtype));
  }

  half* out_ptr = static_cast<half*>(output->GetPtr<void>());
  half* input_ptr = static_cast<half*>(hidden_states->GetPtr<void>());

  void* w1_ptr = w1->GetPtr<void>();
  void* w2_ptr = w2->GetPtr<void>();
  if (w1_ptr == nullptr || w2_ptr == nullptr) {
    KLLM_THROW("w1 or w2 is nullptr.");
  }

  void* w1_scales = w1->scales->GetPtr<void>();
  void* w2_scales = w2->scales->GetPtr<void>();
  if (w1_scales == nullptr || w2_scales == nullptr) {
    KLLM_THROW("w1_scales or w2_scales is nullptr.");
  }

  int* w1_g_idx = (w1->g_idx ? static_cast<int*>(w1->g_idx->GetPtr<void>()) : nullptr);
  int* w2_g_idx = (w2->g_idx ? static_cast<int*>(w2->g_idx->GetPtr<void>()) : nullptr);
  void* w1_zeros = (w1->zeros ? w1->zeros->GetPtr<void>() : nullptr);
  void* w2_zeros = (w2->zeros ? w2->zeros->GetPtr<void>() : nullptr);
  int* w1_perm = (w1->perm ? static_cast<int*>(w1->perm->GetPtr<void>()) : nullptr);
  int* w2_perm = (w2->perm ? static_cast<int*>(w2->perm->GetPtr<void>()) : nullptr);

  half* gating_ptr = static_cast<half*>(gating_output->GetPtr<void>());
  float* gating_float = static_cast<float*>(workspace_buffer_->GetPtr<void>());
  int gating_size = static_cast<int>(gating_output->shape[0] * gating_output->shape[1]);

  DataToFloat<half>(gating_ptr, gating_size, 1, 1, gating_float, context_->device->GetPrimaryComputeStream().Get());

  void* ws = static_cast<char*>(workspace_buffer_->GetPtr<void>()) + max_gating_size_;
  size_t ws_size = max_ws_bytes_ - max_gating_size_;
  int num_tokens = static_cast<int>(hidden_states->shape[0]);

  llm_kernels::nvidia::MOEExpertScaleNormalizationMode norm_mode =
      static_cast<llm_kernels::nvidia::MOEExpertScaleNormalizationMode>(static_cast<int>(moe_scale_norm_mode_));
  FusedMarlinMoe(out_ptr, input_ptr, gating_float, w1_ptr, w2_ptr, w1_scales, w2_scales, ws, ws_size, num_tokens,
                 expert_inter_size_, expert_hidden_size_, expert_num_, expert_topk_,
                 context_->device->GetPrimaryComputeStream().Get(), norm_mode, w1_g_idx, w2_g_idx, w1_perm, w2_perm,
                 w1_zeros, w2_zeros, num_bits_, group_size_);
  output->shape = hidden_states->shape;
  output->dtype = hidden_states->dtype;
  return Status();
}

}  // namespace ksana_llm
