/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/dspark_markov_head.h"

namespace ksana_llm {

DSparkMarkovHead::DSparkMarkovHead(const std::string& w1_weight_name, const std::string& w2_weight_name,
                                   LayerCreationContext& creation_context,
                                   const LinearComputeBackend& linear_compute_backend)
    : context_(creation_context.context) {
  markov_w1_weight_ = creation_context.base_weight->GetModelWeights(w1_weight_name);
  markov_w2_ = std::make_unique<Linear>(w2_weight_name, creation_context, linear_compute_backend);
  dspark_layer_ = std::make_unique<DSparkLayer>();
  dspark_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

Status DSparkMarkovHead::Forward(const uint32_t* token_ids_dev, size_t batch_size, Tensor& logits_row,
                                 std::vector<Tensor>& embed_tensors, std::vector<Tensor>& bias_tensors) {
  // markov_w1 lookup (full [V, R] replicated) -> embed_tensors[0]: [B, R].
  STATUS_CHECK_RETURN(ForwardEmbed(token_ids_dev, batch_size, embed_tensors[0]));

  // markov_w2 matmul (full [R, V] replicated) -> bias_tensors[0]: [B, V].
  STATUS_CHECK_RETURN(markov_w2_->Forward(embed_tensors[0], bias_tensors[0]));

  // logits_row (fp32) += bias (bf16/fp16), mixed-precision.
  STATUS_CHECK_RETURN(dspark_layer_->ForwardLogitsAdd(logits_row.GetPtr<float>(), bias_tensors[0].GetPtr<void>(),
                                                      logits_row.shape[0], logits_row.shape[1], bias_tensors[0].dtype));
  return Status();
}

Status DSparkMarkovHead::ForwardEmbed(const uint32_t* token_ids_dev, size_t batch_size, Tensor& embed_out) {
  const size_t markov_rank = markov_w1_weight_.shape[1];

  STATUS_CHECK_RETURN(dspark_layer_->ForwardMarkovEmbedLookup(embed_out.GetPtr<void>(), token_ids_dev,
                                                              markov_w1_weight_.GetPtr<void>(), batch_size, markov_rank,
                                                              markov_w1_weight_.dtype));

  embed_out.shape = {batch_size, markov_rank};
  embed_out.dtype = markov_w1_weight_.dtype;
  return Status();
}

}  // namespace ksana_llm
