/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/moe_layer.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {
#ifdef ENABLE_CUDA
template <typename T>
Status MoeLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;

  int parameter_index = 0;
  moe_scale_norm_mode_ = std::any_cast<const MoeScaleNormMode>(parameters[parameter_index++]);
  max_token_num_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_num_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_hidden_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_inter_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_topk_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  tp_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  return Status();
}

template <typename T>
size_t MoeLayer<T>::GetWorkSpaceSize() {
  GetMoeGemmWorkspaceSize<T>(max_token_num_, expert_num_, expert_hidden_size_, expert_inter_size_, expert_topk_,
                             tp_size_, rank_, use_lora_, max_ws_bytes_);
  return max_ws_bytes_;
}

template <typename T>
Status MoeLayer<T>::SetWorkSpaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) {
  workspace_buffer_ = workspace_buffer;
  size_t scale_probabilities_size = max_token_num_ * expert_num_ * sizeof(float);
  size_t src_to_dest_map_size = expert_topk_ * max_token_num_ * sizeof(int);
  size_t selected_expert_size = expert_topk_ * max_token_num_ * sizeof(int);
  size_t lora_workspace_size = 0;  // NO support for lora
  size_t moe_workspace_size =
      max_ws_bytes_ - scale_probabilities_size - src_to_dest_map_size - selected_expert_size - lora_workspace_size;

  void* ws = workspace_buffer_->GetPtr<void>();
  if (ws) {
    workspace_info_.size = max_ws_bytes_;
    workspace_info_.workspace = ws;
    workspace_info_.scale_probs =
        llm_kernels::utils::nextWorkspacePtr(reinterpret_cast<int8_t*>(workspace_info_.workspace), moe_workspace_size);
    workspace_info_.src_to_dest_map = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(workspace_info_.scale_probs), scale_probabilities_size);
    workspace_info_.selected_experts = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(workspace_info_.src_to_dest_map), src_to_dest_map_size);
    workspace_info_.lora_workspace = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(workspace_info_.selected_experts), selected_expert_size);
  }
  return Status();
}

template <typename T>
Status MoeLayer<T>::Preprocess(const ModelConfig& model_config_) {
  config_map_.resize(model_config_.max_batch_size + 1);
  for (size_t m = 1; m <= static_cast<size_t>(model_config_.max_batch_size); m++) {
    size_t best_config_index = InvokeMoeGemmConfigProfile<T>();
    config_map_[m] = best_config_index;
  }
  return Status();
}

template <typename T>
Status MoeLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  const size_t num_tokens = input_tensors[0].shape[0];
  size_t best_config_index = 0;  // TODO(winminkong): op optimization

  // input_tensors: 0.hidden states 1.routing_out 2.up_gate_experts 3.down_experts
  if (moe_scale_norm_mode_ == MoeScaleNormMode::RE_NORM) {
    InvokeMoeCutlassGemm<T, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE>(
        input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
        input_tensors[3].GetPtr<void>(), num_tokens, expert_hidden_size_, expert_inter_size_, expert_num_, expert_topk_,
        static_cast<char*>(workspace_info_.workspace), output_tensors[0].GetPtr<void>(), workspace_info_.scale_probs,
        static_cast<int*>(workspace_info_.src_to_dest_map), static_cast<int*>(workspace_info_.selected_experts),
        tp_size_, rank_, use_lora_, best_config_index, context_->GetComputeStreams()[rank_].Get());
  } else if (moe_scale_norm_mode_ == MoeScaleNormMode::NO_NORM) {
    InvokeMoeCutlassGemm<T, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE>(
        input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
        input_tensors[3].GetPtr<void>(), num_tokens, expert_hidden_size_, expert_inter_size_, expert_num_, expert_topk_,
        static_cast<char*>(workspace_info_.workspace), output_tensors[0].GetPtr<void>(), workspace_info_.scale_probs,
        static_cast<int*>(workspace_info_.src_to_dest_map), static_cast<int*>(workspace_info_.selected_experts),
        tp_size_, rank_, use_lora_, best_config_index, context_->GetComputeStreams()[rank_].Get());
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}

template class MoeLayer<float>;
template class MoeLayer<half>;
#  ifdef ENABLE_BFLOAT16
template class MoeLayer<__nv_bfloat16>;
#  endif

#endif
}  // namespace ksana_llm
