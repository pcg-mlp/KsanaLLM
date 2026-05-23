/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/layernorm.h"

#include "layers/layernorm_layer.h"

namespace ksana_llm {

Layernorm::Layernorm(const std::string& weight_name, float layernorm_eps, const LayerCreationContext& creation_context,
                     const std::string& weight_bias_name) {
  layernorm_layer_ = std::make_shared<LayernormLayer>();
  layernorm_layer_->Init({layernorm_eps}, creation_context.runtime_config, creation_context.context,
                         creation_context.rank);
  weight_ = creation_context.base_weight->GetModelWeights(weight_name);
  if (weight_bias_name != "") {
    weight_bias_ = creation_context.base_weight->GetModelWeights(weight_bias_name);
    with_bias_ = true;
  }
}

Layernorm::Layernorm(const std::string& weight_name, const std::string& weight2_name, float layernorm_eps,
                     const LayerCreationContext& creation_context, int32_t stride) {
  layernorm_layer_ = std::make_shared<LayernormLayer>();
  layernorm_layer_->Init({layernorm_eps}, creation_context.runtime_config, creation_context.context,
                         creation_context.rank);
  weight_ = creation_context.base_weight->GetModelWeights(weight_name);
  weight2_ = creation_context.base_weight->GetModelWeights(weight2_name);
  stride_ = stride;
}

Layernorm::Layernorm(float layernorm_eps, uint32_t num_heads, uint32_t head_dim,
                     const LayerCreationContext& creation_context) {
  with_weight_ = false;
  layernorm_layer_ = std::make_shared<LayernormLayer>();
  layernorm_layer_->Init({layernorm_eps, num_heads, head_dim}, creation_context.runtime_config,
                         creation_context.context, creation_context.rank);
}

Layernorm::~Layernorm() {}

Status Layernorm::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK_WITH_INFO(input_tensors.size() == 1ul, "Layernorm::Forward requires 1 input tensor");
  if (!with_weight_) {
    // No weight per-head RMSNorm
    return layernorm_layer_->Forward({input_tensors[0]}, output_tensors);
  } else if (with_bias_) {
    // With bias weight LayerNorm
    return layernorm_layer_->Forward({input_tensors[0], weight_, weight_bias_}, output_tensors);
  } else {
    // No bias weight RMSNorm
    return layernorm_layer_->Forward({input_tensors[0], weight_}, output_tensors);
  }
}

Status Layernorm::ForwardParallel(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK_WITH_INFO(input_tensors.size() == 2ul, "Layernorm::ForwardParallel requires 2 input tensors");
  Tensor in1 = input_tensors[0];
  Tensor in2 = input_tensors[1];
  if (stride_ > 0) {
    // Encode the row stride as a 3rd shape dimension so the layer can extract it.
    in1.shape = {in1.shape[0], in1.shape[1], static_cast<size_t>(stride_)};
    in2.shape = {in2.shape[0], in2.shape[1], static_cast<size_t>(stride_)};
  }
  return layernorm_layer_->Forward({in1, weight_, in2, weight2_}, output_tensors);
}

}  // namespace ksana_llm
