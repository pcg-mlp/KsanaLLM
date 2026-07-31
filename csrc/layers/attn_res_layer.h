/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

// AttnResLayer: fused attention-residual aggregation.
//
// For each token, scores the current prefix against a bank of frozen prior
// residual rows (RMSNorm(row) * qk_weight, dot product), softmax-mixes them,
// and applies an optional output RMSNorm.
// When `delta` is non-null, `prefix` is updated in place to prefix + delta.
// When `write_block` is true, the (updated) prefix is snapshotted into
// blocks[:, num_blocks, :] (the bank row indexed by the current num_blocks).
class AttnResLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  // prefix             [num_tokens, hidden_size]             in-place updated if delta != nullptr
  // delta              [num_tokens, hidden_size]             pending residual add, or nullptr
  // blocks             [num_tokens, block_num, hidden_size]  bank of frozen residual rows
  // norm_weight        [hidden_size]                         score RMSNorm weight
  // qk_weight          [hidden_size]                         score projection weight
  // output_norm_weight [hidden_size]                         output RMSNorm weight, or nullptr
  // output             [num_tokens, hidden_size]             aggregated (optionally normed) result
  // write_block        bool                                  write updated prefix into blocks[:, num_blocks, :]
  Status Forward(const Tensor* prefix, const Tensor* delta, const Tensor* blocks, const Tensor* norm_weight,
                 const Tensor* qk_weight, const Tensor* output_norm_weight, Tensor* output, bool write_block,
                 const LayerExecContext& exec_ctx = {});

 private:
  template <typename T>
  Status ForwardT(const Tensor& prefix, const Tensor* delta, const Tensor& blocks, const Tensor& norm_weight,
                  const Tensor& qk_weight, const Tensor* output_norm_weight, Tensor& output, bool write_block);

 private:
  size_t hidden_dim_;
};

}  // namespace ksana_llm
