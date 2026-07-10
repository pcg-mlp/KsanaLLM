/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <string>

#include "layers/dspark_layer.h"
#include "models/base/layer_creation_context.h"
#include "models/base/model_communicator.h"
#include "modules/basic/linear.h"

namespace ksana_llm {

// DSpark Markov head: additive logits-bias generator on the DSpark serial path.
//
// TP layout (hidden-parallel W1, vocab-parallel W2):
//   * markov_w1.weight  shape [V, R/TP]
//   * markov_w2.weight  shape [V/TP, R]
class DSparkMarkovHead {
 public:
  DSparkMarkovHead(const std::string& w1_weight_name, const std::string& w2_weight_name,
                   LayerCreationContext& creation_context, const LinearComputeBackend& linear_compute_backend,
                   std::shared_ptr<ModelCommunicator> model_comm);

  ~DSparkMarkovHead() = default;

  // Full markov bias correction: embed lookup → AllGather → w2 matmul → AllGather → logits += bias.
  Status Forward(const Tensor& token_ids, Tensor& logits_row, std::vector<Tensor>& embed_tensors,
                 std::vector<Tensor>& bias_tensors);

 private:
  // markov_w1 lookup — TP-local.
  Status ForwardEmbed(const Tensor& token_ids, Tensor& embed_shard_out);

  std::shared_ptr<Context> context_;
  std::shared_ptr<ModelCommunicator> model_comm_;

  Tensor markov_w1_weight_;
  std::unique_ptr<Linear> markov_w2_;
  std::unique_ptr<DSparkLayer> dspark_layer_;
};

}  // namespace ksana_llm
