/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "models/base/layer_creation_context.h"

namespace ksana_llm {

class Layernorm {
 public:
  // Disable a default constructor
  Layernorm(const std::string& weight_name, float layernorm_eps, const LayerCreationContext& creation_context,
            const std::string& weight_bias_name = "");

  // Constructor with two weights for parallel norm.
  Layernorm(const std::string& weight_name, const std::string& weight2_name, float layernorm_eps,
            const LayerCreationContext& creation_context, int32_t stride = 0);

  Layernorm(float layernorm_eps, uint32_t num_heads, uint32_t head_dim, const LayerCreationContext& creation_context);

  ~Layernorm();

  Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // Parallelly computes norm on two inputs with different dims
  Status ForwardParallel(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> layernorm_layer_;
  Tensor weight_;
  Tensor weight2_;
  Tensor weight_bias_;
  bool with_weight_ = true;
  bool with_bias_ = false;
  int32_t stride_{0};  // Row stride (in elements) for parallel norm strided inputs; 0 = contiguous
};

}  // namespace ksana_llm
