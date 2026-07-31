/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <string>

#include "models/base/layer_creation_context.h"

namespace ksana_llm {

class AttnRes {
 public:
  AttnRes(const std::string& weight_name, const LayerCreationContext& creation_context,
          const std::string& output_norm_weight_name = "");

  ~AttnRes() = default;

  // Aggregate one attention-residual step.
  // input_tensors[0]:  prefix  [num_tokens, hidden_size]                   (in-place updated if delta given)
  // input_tensors[1]:  blocks  [num_tokens, num_valid_blocks, hidden_size] bank slice
  // input_tensors[2]:  delta   [num_tokens, hidden_size]                   optional
  // output_tensors[0]: output  [num_tokens, hidden_size]                   aggregated (optionally normed) result
  // write_block: if true, snapshot the (updated) prefix into blocks[:, shape[1], :]
  //              (i.e. the first row past the current valid range).
  Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors, bool write_block);

 private:
  Tensor norm_weight_;         // [hidden_size]  score RMSNorm weight
  Tensor proj_weight_;         // [hidden_size]  score projection weight
  Tensor output_norm_weight_;  // [hidden_size]  output RMSNorm weight (optional)

  std::unique_ptr<BaseLayer> attn_res_layer_;
};

}  // namespace ksana_llm
