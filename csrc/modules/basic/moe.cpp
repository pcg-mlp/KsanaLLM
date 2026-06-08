/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/moe.h"

#include "layers/base_moe_layer.h"

namespace ksana_llm {

MoE::MoE(const int& layer_idx, const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
         const LayerCreationContext& creation_context, MoeScaleNormMode moe_scale_norm_mode)
    : use_e_score_correction_bias_(false), layer_idx_(layer_idx) {
  Init(up_gate_proj_weight_name, down_proj_weight_name, creation_context, moe_scale_norm_mode);
}

MoE::MoE(const int& layer_idx, const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
         const std::string& e_score_correction_bias_weight_name, const LayerCreationContext& creation_context,
         MoeScaleNormMode moe_scale_norm_mode)
    : use_e_score_correction_bias_(true), layer_idx_(layer_idx) {
  Init(up_gate_proj_weight_name, down_proj_weight_name, creation_context, moe_scale_norm_mode);
  e_score_correction_bias_weight_ = creation_context.base_weight->GetModelWeights(e_score_correction_bias_weight_name);
}

void MoE::Init(const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
               const LayerCreationContext& creation_context, MoeScaleNormMode moe_scale_norm_mode) {
  moe_layer_ = creation_context.moe_layer_factory->AutoCreateMoeLayer(
      creation_context.base_weight, std::vector<std::string>{up_gate_proj_weight_name, down_proj_weight_name},
      creation_context.weight_type, creation_context.input_type, creation_context.output_type, layer_idx_,
      {moe_scale_norm_mode});

  moe_layer_->SetWorkspaceBuffer(creation_context.workspace_mgr->GetWorkspace(moe_layer_->GetWorkspaceSize()));
  moe_layer_->Preprocess(creation_context.model_config, creation_context.runtime_config);

  up_gate_proj_weight_ = creation_context.base_weight->GetModelWeights(up_gate_proj_weight_name);
  down_proj_weight_ = creation_context.base_weight->GetModelWeights(down_proj_weight_name);
  eplb_expert_map_ = creation_context.base_weight->GetModelWeights("expert_map");
}

MoE::~MoE() {}

Status MoE::Forward(Tensor hidden_states, Tensor gating_output, Tensor workspace_tensor,
                    std::vector<Tensor>& output_tensors, const Tensor* input_ids) {
  const Tensor* e_score_bias_ptr = use_e_score_correction_bias_ ? &e_score_correction_bias_weight_ : nullptr;
  Tensor output = output_tensors[0];
  STATUS_CHECK_RETURN(moe_layer_->Forward(&hidden_states, &gating_output, &up_gate_proj_weight_, &down_proj_weight_,
                                          &eplb_expert_map_, &output, &workspace_tensor, e_score_bias_ptr, input_ids));
  output_tensors[0].shape = output.shape;
  return Status();
}

}  // namespace ksana_llm
