/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/attn_res.h"

#include "layers/attn_res_layer.h"

namespace ksana_llm {

AttnRes::AttnRes(const std::string& weight_name, const LayerCreationContext& creation_context,
                 const std::string& output_norm_weight_name) {
  attn_res_layer_ = std::make_unique<AttnResLayer>();
  attn_res_layer_->Init({creation_context.model_config.layernorm_eps, creation_context.model_config.layernorm_eps,
                         creation_context.model_config.hidden_units},
                        creation_context.runtime_config, creation_context.context, creation_context.rank);
  norm_weight_ = creation_context.base_weight->GetModelWeights(weight_name + "_norm.weight");
  proj_weight_ = creation_context.base_weight->GetModelWeights(weight_name + "_proj.weight");
  if (!output_norm_weight_name.empty()) {
    output_norm_weight_ = creation_context.base_weight->GetModelWeights(output_norm_weight_name);
  }
}

Status AttnRes::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                        bool write_block) {
  KLLM_CHECK(input_tensors.size() == 2 || input_tensors.size() == 3);
  KLLM_CHECK(output_tensors.size() == 1);

  PROFILE_EVENT_SCOPE(AttnRes_Forward, fmt::format("AttnRes_Forward_seq{}", input_tensors[0].shape[0]));

  const Tensor& prefix = input_tensors[0];
  const Tensor& blocks = input_tensors[1];
  const Tensor* delta_ptr = (input_tensors.size() == 3) ? &input_tensors[2] : nullptr;
  const Tensor* output_norm_weight_ptr = (output_norm_weight_.GetElementNumber() > 0) ? &output_norm_weight_ : nullptr;

  auto* typed = static_cast<AttnResLayer*>(attn_res_layer_.get());
  return typed->Forward(&prefix, delta_ptr, &blocks, &norm_weight_, &proj_weight_, output_norm_weight_ptr,
                        &output_tensors[0], write_block);
}

}  // namespace ksana_llm
