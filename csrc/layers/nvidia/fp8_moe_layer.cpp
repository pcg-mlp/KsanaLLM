/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include "layers/fp8_moe_layer.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"
#include "kernels/nvidia/kernel_wrapper.h"
#include "utils/utils.h"

namespace ksana_llm {
#define WT fp8e4m3  // TODO(robertyuan): Only support fp8e4m3 rightnow.

Status Fp8MoeLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                         std::shared_ptr<Context> context, int rank) {
  return MoeLayer::Init(parameters, runtime_config, context, rank);
}

size_t Fp8MoeLayer::GetWorkspaceSize() { DISPATCH_BY_2_DTYPE(inter_data_type_, GetWorkspaceSizeT); }

Status Fp8MoeLayer::Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) {
  DISPATCH_BY_2_DTYPE(inter_data_type_, PreprocessT, model_config, runtime_config);
}

template <typename T>
size_t Fp8MoeLayer::GetWorkspaceSizeT() {
  GetMoeGemmWorkspaceSize<WT, WT, T>(this->params_.max_token_num, this->params_.expert_num_per_node,
                                     this->params_.expert_hidden_size, this->params_.expert_inter_size,
                                     this->params_.expert_topk, this->params_.tp_size, this->rank_, this->use_lora_,
                                     this->max_ws_bytes_, this->workspace_info_.workspace_sizes);
  quant_buffer_size_ = this->params_.max_token_num * this->params_.expert_hidden_size * GetTypeSize(TYPE_FP8_E4M3);
  this->max_ws_bytes_ += quant_buffer_size_;
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for Fp8MoeLayer", this->rank_, this->max_ws_bytes_);
  return this->max_ws_bytes_;
}

Status Fp8MoeLayer::SetWorkspaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) {
  this->workspace_buffer_ = workspace_buffer;
  this->scale_probabilities_size_ = this->params_.max_token_num * this->params_.expert_num_per_node * sizeof(float);
  this->src_to_dest_map_size_ = this->params_.expert_topk * this->params_.max_token_num * sizeof(int);
  this->selected_expert_size_ = this->params_.expert_topk * this->params_.max_token_num * sizeof(int);
  this->lora_workspace_size_ = 0;  // NO support for lora
  this->moe_workspace_size_ = this->max_ws_bytes_ - this->scale_probabilities_size_ - this->src_to_dest_map_size_ -
                              this->selected_expert_size_ - this->lora_workspace_size_ - quant_buffer_size_;
  return Status();
}

template <typename T>
Status Fp8MoeLayer::PreprocessT(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) {
  bool is_fp8 = true;
  best_config_index_ = InvokeMoeGemmConfigProfile<WT, WT, T>(this->tactics_, is_fp8);
  return Status();
}

template <typename T>
Status Fp8MoeLayer::ForwardT(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1,
                             const Tensor* w2, const Tensor* eplb_map, Tensor* output, const Tensor* e_score_bias,
                             const Tensor* input_ids) {
  const size_t num_tokens = hidden_states->shape[0];
  bool is_fp8 = true;
  void* fc1_dequant_scale = nullptr;
  void* fc2_quant_scale = nullptr;
  void* fc2_dequant_scale = nullptr;
  std::shared_ptr<Tensor> buffer = this->workspace_buffer_;
  void* input_activations = buffer->GetPtr<void>();

  if (this->set_workspace_buffer_info_) {
    this->set_workspace_buffer_info_ = false;

    this->workspace_info_.size = this->max_ws_bytes_ - quant_buffer_size_;
    this->workspace_info_.workspace = reinterpret_cast<uint8_t*>(buffer->GetPtr<void>()) + quant_buffer_size_;
    this->workspace_info_.scale_probs = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(this->workspace_info_.workspace), this->moe_workspace_size_);
    this->workspace_info_.src_to_dest_map = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(this->workspace_info_.scale_probs), this->scale_probabilities_size_);
    this->workspace_info_.selected_experts = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(this->workspace_info_.src_to_dest_map), this->src_to_dest_map_size_);
    this->workspace_info_.lora_workspace = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(this->workspace_info_.selected_experts), this->selected_expert_size_);
  }

  if (w1->weight_scales && w1->input_scales) {
    float* fc1_quant_scale = static_cast<float*>(w1->input_scales->GetPtr<void>());
    Fp8E4m3Quantize<T>(1, hidden_states->GetElementNumber(), static_cast<T*>(hidden_states->GetPtr<void>()),
                       input_activations, fc1_quant_scale, true,
                       this->context_->device->GetPrimaryComputeStream().Get());
    fc1_dequant_scale = w1->weight_scales->GetPtr<void>();
    fc2_quant_scale = w2->input_scales->GetPtr<void>();
    fc2_dequant_scale = w2->weight_scales->GetPtr<void>();
  } else {
    KLLM_THROW("Unsupported fp8 moe method, only support both weight_scale and input_scale are not null.");
  }
  void* e_score_correction_bias_weight_void = nullptr;
  if (this->params_.use_e_score_correction_bias) {
    e_score_correction_bias_weight_void = eplb_map->GetPtr<void>();
  }
  if (this->params_.moe_scale_norm_mode == MoeScaleNormMode::RE_NORM) {
    InvokeMoeCutlassGemm<WT, WT, T, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE>(
        input_activations, gating_output->GetPtr<void>(), w1->GetPtr<void>(), w2->GetPtr<void>(),
        e_score_correction_bias_weight_void, num_tokens, this->params_.expert_hidden_size,
        this->params_.expert_inter_size, this->params_.expert_num_per_node, this->params_.expert_topk,
        this->workspace_info_.workspace_sizes, static_cast<char*>(this->workspace_info_.workspace),
        output->GetPtr<void>(), this->workspace_info_.scale_probs,
        static_cast<int*>(this->workspace_info_.src_to_dest_map),
        static_cast<int*>(this->workspace_info_.selected_experts), this->params_.tp_size, this->rank_, this->use_lora_,
        best_config_index_, this->tactics_, this->params_.use_vllm_moe, this->params_.num_expert_group,
        this->params_.expert_groups_topk, this->params_.scoring_func, this->params_.topk_method,
        this->params_.norm_topk_prob, this->params_.routed_scaling_factor, this->params_.use_e_score_correction_bias,
        this->context_->device->GetPrimaryComputeStream().Get(), is_fp8, fc1_dequant_scale, fc2_quant_scale,
        fc2_dequant_scale);
  } else if (this->params_.moe_scale_norm_mode == MoeScaleNormMode::NO_NORM) {
    InvokeMoeCutlassGemm<WT, WT, T, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE>(
        input_activations, gating_output->GetPtr<void>(), w1->GetPtr<void>(), w2->GetPtr<void>(),
        e_score_correction_bias_weight_void, num_tokens, this->params_.expert_hidden_size,
        this->params_.expert_inter_size, this->params_.expert_num_per_node, this->params_.expert_topk,
        this->workspace_info_.workspace_sizes, static_cast<char*>(this->workspace_info_.workspace),
        output->GetPtr<void>(), this->workspace_info_.scale_probs,
        static_cast<int*>(this->workspace_info_.src_to_dest_map),
        static_cast<int*>(this->workspace_info_.selected_experts), this->params_.tp_size, this->rank_, this->use_lora_,
        best_config_index_, this->tactics_, this->params_.use_vllm_moe, this->params_.num_expert_group,
        this->params_.expert_groups_topk, this->params_.scoring_func, this->params_.topk_method,
        this->params_.norm_topk_prob, this->params_.routed_scaling_factor, this->params_.use_e_score_correction_bias,
        this->context_->device->GetPrimaryComputeStream().Get(), is_fp8, fc1_dequant_scale, fc2_quant_scale,
        fc2_dequant_scale);
  }

  output->shape = hidden_states->shape;
  output->dtype = hidden_states->dtype;
  return Status();
}

#undef WT

Status Fp8MoeLayer::Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1,
                            const Tensor* w2, const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                            const Tensor* e_score_bias, const Tensor* input_ids, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_2_DTYPE(inter_data_type_, ForwardT, hidden_states, gating_output, w1, w2, eplb_map, output, e_score_bias,
                      input_ids);
}

}  // namespace ksana_llm
