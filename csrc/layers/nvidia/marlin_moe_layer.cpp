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

  params_ = std::any_cast<MoeLayerParameters>(parameters.front());

  const float swiglu_limit = params_.swiglu_limit;
  KLLM_CHECK_WITH_INFO(!(swiglu_limit > 0.0f), "MarlinMoeLayer does not support swiglu_limit.");
  return Status();
}

size_t MarlinMoeLayer::GetWorkspaceSize() { DISPATCH_BY_3_DTYPE(inter_data_type_, GetWorkspaceSizeT); }

template <typename T>
size_t MarlinMoeLayer::GetWorkspaceSizeT() {
  max_gating_size_ = sizeof(float) * params_.max_token_num * params_.expert_num_per_node;
  max_ws_bytes_ = max_gating_size_;
  max_ws_bytes_ +=
      InvokeGetFusedMarlinMoeWorkspaceSize(params_.max_token_num, params_.expert_inter_size, params_.expert_hidden_size,
                                           params_.expert_num_per_node, params_.expert_topk, sizeof(T));
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
      static_cast<llm_kernels::nvidia::MOEExpertScaleNormalizationMode>(static_cast<int>(params_.moe_scale_norm_mode));
  FusedMarlinMoe(out_ptr, input_ptr, gating_float, w1_ptr, w2_ptr, w1_scales, w2_scales, ws, ws_size, num_tokens,
                 params_.expert_inter_size, params_.expert_hidden_size, params_.expert_num_per_node,
                 params_.expert_topk, context_->device->GetPrimaryComputeStream().Get(), norm_mode, w1_g_idx, w2_g_idx,
                 w1_perm, w2_perm, w1_zeros, w2_zeros, num_bits_, params_.group_size);
  output->shape = hidden_states->shape;
  output->dtype = hidden_states->dtype;
  return Status();
}

}  // namespace ksana_llm
