/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "layers/base_moe_layer.h"
#include "models/base/layer_creation_context.h"

namespace ksana_llm {

class MoE {
 public:
  // Disable a default constructor
  MoE(const int& layer_idx, const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
      const LayerCreationContext& creation_context, MoeScaleNormMode moe_scale_norm_mode);

  MoE(const int& layer_idx, const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
      const std::string& e_score_correction_bias_weight_name, const LayerCreationContext& creation_context,
      MoeScaleNormMode moe_scale_norm_mode);

  ~MoE();

  // input_ids is optional and used only for DSv4 MoE layers with hash-based
  // routing (TopKSoftplusSqrtLayer in hash mode). Pass nullptr otherwise.
  Status Forward(Tensor hidden_states, Tensor gating_output, Tensor workspace_tensor,
                 std::vector<Tensor>& output_tensors, const Tensor* input_ids = nullptr);

  // Accessors for MoE precision dump helpers; returns the per-rank loaded weight tensors.
  const Tensor& GetUpGateProjWeight() const { return up_gate_proj_weight_; }
  const Tensor& GetDownProjWeight() const { return down_proj_weight_; }

 private:
  void Init(const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
            const LayerCreationContext& creation_context, MoeScaleNormMode moe_scale_norm_mode);

 protected:
  int layer_idx_;

  std::shared_ptr<BaseMoeLayer> moe_layer_;
  Tensor up_gate_proj_weight_;
  Tensor down_proj_weight_;
  Tensor eplb_expert_map_;

  bool use_e_score_correction_bias_;
  // Despite the name, this "gate extra weight" slot holds different tensors per model:
  //   - V3 / GLM4: gate.e_score_correction_bias (fp32 [num_experts])
  //   - DSv4 hash layers: gate.tid2eid (int32 [vocab, topk])
  // MoeLayer disambiguates via scoring_func_ + use_hash_. See MoE::Forward slot-5 comment.
  Tensor e_score_correction_bias_weight_;
};
}  // namespace ksana_llm
