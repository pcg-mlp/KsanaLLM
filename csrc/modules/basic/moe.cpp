/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/moe.h"

#include "layers/moe_layer.h"

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
  std::vector<Tensor> moe_output_tensors = {output_tensors[0], workspace_tensor};
  std::vector<Tensor> moe_input_tensors = {hidden_states, gating_output, up_gate_proj_weight_,
                                           down_proj_weight_, eplb_expert_map_};
  if (use_e_score_correction_bias_) {
    // Slot 5 carries either gate.e_score_correction_bias (fp32) for V3/GLM4
    // or gate.tid2eid (int32) for DSv4 hash layers. MoeLayer decides how to
    // interpret it via scoring_func_ + use_hash_.
    moe_input_tensors.push_back(e_score_correction_bias_weight_);
  }
  if (input_ids != nullptr) {
    // Slot 6: DSv4 hash layers feed input_ids into TopKSoftplusSqrtLayer.
    moe_input_tensors.push_back(*input_ids);
  }
  STATUS_CHECK_RETURN(moe_layer_->Forward(moe_input_tensors, moe_output_tensors));
  output_tensors[0].shape = moe_output_tensors[0].shape;
  return Status();
}

}  // namespace ksana_llm
