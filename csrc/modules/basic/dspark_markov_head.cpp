/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/dspark_markov_head.h"

namespace ksana_llm {

DSparkMarkovHead::DSparkMarkovHead(const std::string& w1_weight_name, const std::string& w2_weight_name,
                                   LayerCreationContext& creation_context,
                                   const LinearComputeBackend& linear_compute_backend,
                                   std::shared_ptr<ModelCommunicator> model_comm)
    : context_(creation_context.context), model_comm_(std::move(model_comm)) {
  markov_w1_weight_ = creation_context.base_weight->GetModelWeights(w1_weight_name);
  markov_w2_ = std::make_unique<Linear>(w2_weight_name, creation_context, linear_compute_backend);
  dspark_layer_ = std::make_unique<DSparkLayer>();
  dspark_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

Status DSparkMarkovHead::Forward(const Tensor& token_ids, Tensor& logits_row, std::vector<Tensor>& embed_tensors,
                                 std::vector<Tensor>& bias_tensors) {
  // markov_w1 lookup (TP-local) -> embed_tensors: [B, R/TP].
  STATUS_CHECK_RETURN(ForwardEmbed(token_ids, embed_tensors[0]));

  // AllGather along R: [B, R/TP] -> [B, R] (when TP > 1).
  if (model_comm_) {
    model_comm_->AllGather({embed_tensors[0], /*permute_buf=*/bias_tensors[0]}, embed_tensors);
  }

  // markov_w2 matmul (TP-local) -> bias_tensors: [B, V/TP].
  STATUS_CHECK_RETURN(markov_w2_->Forward(embed_tensors[0], bias_tensors[0]));

  // AllGather along V: [B, V/TP] -> [B, V] (when TP > 1).
  if (model_comm_) {
    model_comm_->AllGather({bias_tensors[0], /*permute_buf=*/embed_tensors[0]}, bias_tensors);
  }

  // logits_row (fp32) += bias (bf16/fp16), mixed-precision.
  STATUS_CHECK_RETURN(dspark_layer_->ForwardLogitsAdd(logits_row.GetPtr<float>(), bias_tensors[0].GetPtr<void>(),
                                                      logits_row.shape[0], logits_row.shape[1], bias_tensors[0].dtype));
  return Status();
}

Status DSparkMarkovHead::ForwardEmbed(const Tensor& token_ids, Tensor& embed_shard_out) {
  const size_t batch_size = token_ids.shape[0];
  const int vocab_size = static_cast<int>(markov_w1_weight_.shape[0]);
  const size_t markov_rank = markov_w1_weight_.shape[1];

  STATUS_CHECK_RETURN(dspark_layer_->ForwardMarkovEmbedLookup(
      embed_shard_out.GetPtr<void>(), token_ids.GetPtr<int32_t>(), markov_w1_weight_.GetPtr<void>(), batch_size,
      markov_rank, vocab_size, markov_w1_weight_.dtype));

  embed_shard_out.shape = {batch_size, markov_rank};
  embed_shard_out.dtype = markov_w1_weight_.dtype;
  return Status();
}

}  // namespace ksana_llm
