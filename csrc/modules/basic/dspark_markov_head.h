/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <string>

#include "layers/dspark_layer.h"
#include "models/base/layer_creation_context.h"
#include "modules/basic/linear.h"

namespace ksana_llm {

// DSpark Markov head: additive logits-bias generator on the DSpark serial path.
//
// Weights are replicated on every TP rank (no TP split):
//   * markov_w1.weight  shape [V, R]
//   * markov_w2.weight  shape [R, V]
class DSparkMarkovHead {
 public:
  DSparkMarkovHead(const std::string& w1_weight_name, const std::string& w2_weight_name,
                   LayerCreationContext& creation_context, const LinearComputeBackend& linear_compute_backend);

  ~DSparkMarkovHead() = default;

  // Full markov bias correction: embed lookup -> w2 matmul -> logits += bias.
  Status Forward(const uint32_t* token_ids_dev, size_t batch_size, Tensor& logits_row,
                 std::vector<Tensor>& embed_tensors, std::vector<Tensor>& bias_tensors);

 private:
  // markov_w1 lookup
  Status ForwardEmbed(const uint32_t* token_ids_dev, size_t batch_size, Tensor& embed_out);

  std::shared_ptr<Context> context_;

  Tensor markov_w1_weight_;
  std::unique_ptr<Linear> markov_w2_;
  std::unique_ptr<DSparkLayer> dspark_layer_;
};

}  // namespace ksana_llm
