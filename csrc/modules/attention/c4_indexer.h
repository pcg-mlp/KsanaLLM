/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "layers/base_layer.h"
#include "models/base/forwarding_context.h"
#include "models/base/layer_creation_context.h"
#include "modules/attention/compressor.h"
#include "modules/basic/linear.h"

namespace ksana_llm {

// C4 Indexer module for DeepSeek V4 topk selector with compress-4 key compression.
class C4Indexer {
 public:
  C4Indexer(int layer_idx, const LayerCreationContext& creation_context,
            const ModelCreationConfig& model_creation_config);

  ~C4Indexer() = default;

  // Full forward pass:
  //   1. kv_compressed = compressor_(x)
  //   2. weights = weights_proj_(x)
  //   3. q = wq_b_(qr)
  //   4. topk_indices = indexer(q, kv_compressed, weights)
  // Inputs:
  //   - x: hidden states [total_tokens, hidden_dim]
  //   - qr: query lora output [total_tokens, q_lora_rank]
  //   - workspace: intermediate buffer for compressor and indexer
  // Outputs:
  //   - q_indexer_tensors: qr projected by wq_b [total_tokens, n_heads, head_dim]
  //   - topk_indices: top-k block indices [total_tokens, index_topk]
  Status Forward(const Tensor& x, const Tensor& qr, const Tensor& workspace, std::vector<Tensor>& q_indexer_tensors,
                 Tensor& topk_indices, ForwardingContext& forwarding_context);

 private:
  const int layer_idx_;

  // Model parameters
  size_t n_heads_;     // number of index heads (dsa_config.index_n_heads)
  size_t head_dim_;    // index head dimension (dsa_config.index_head_dim)
  size_t index_topk_;  // top-k value (dsa_config.index_topk)
  size_t block_size_;  // block size for KV cache

  static constexpr size_t kC4Ratio = 4;

  // Linear layers
  std::unique_ptr<Linear> wq_b_;          // Query projection (q_lora_rank -> n_heads * head_dim)
  std::unique_ptr<Linear> weights_proj_;  // Weights projection (dim -> n_heads)

  // Compressor for key compression (is_in_indexer=true, compress_ratio=4)
  std::unique_ptr<Compressor> compressor_;

  // Flash and paged indexer layers for top-k block selection
  std::unique_ptr<BaseLayer> flash_sparse_mla_indexer_layer_;
  std::unique_ptr<BaseLayer> paged_sparse_mla_indexer_layer_;
};

}  // namespace ksana_llm
