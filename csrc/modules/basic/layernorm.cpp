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

Status Layernorm::Forward(const Tensor& input, Tensor& output) {
  auto* typed = static_cast<LayernormLayer*>(layernorm_layer_.get());
  if (!with_weight_) {
    return typed->Forward(&input, &output);
  } else if (with_bias_) {
    return typed->Forward(&input, &weight_, &output, &weight_bias_);
  } else {
    return typed->Forward(&input, &weight_, &output);
  }
}

Status Layernorm::ForwardParallel(const Tensor& in1, const Tensor& in2, Tensor& out1, Tensor& out2) {
  Tensor t1 = in1;
  Tensor t2 = in2;
  if (stride_ > 0) {
    t1.shape = {t1.shape[0], t1.shape[1], static_cast<size_t>(stride_)};
    t2.shape = {t2.shape[0], t2.shape[1], static_cast<size_t>(stride_)};
  }
  auto* typed = static_cast<LayernormLayer*>(layernorm_layer_.get());
  int s1 = (t1.shape.size() > 2) ? static_cast<int>(t1.shape[2]) : 0;
  int s2 = (t2.shape.size() > 2) ? static_cast<int>(t2.shape[2]) : 0;
  if (s1 > 0) t1.shape.resize(2);
  if (s2 > 0) t2.shape.resize(2);
  return typed->Forward(&t1, &weight_, &t2, &weight2_, &out1, &out2, s1, s2);
}

}  // namespace ksana_llm
