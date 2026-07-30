/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "layers/base_layer.h"
#include "models/base/forwarding_context.h"
#include "models/base/layer_creation_context.h"
#include "modules/basic/linear.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

// Forward declarations
class FlashSparseMlaIndexer;
class PagedSparseMlaIndexer;

// Sparse MLA Indexer module
// Implements the indexer logic for sparse multi-head latent attention
class SparseMlaIndexer {
 public:
  SparseMlaIndexer(int layer_idx, LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config);

  ~SparseMlaIndexer() = default;

  // Forward pass
  // Inputs:
  //   - x: hidden states [batch, seq_len, hidden_dim]
  //   - qr: query lora output [batch, seq_len, q_lora_rank]
  //   - workspace: store intermediate results
  //   - context_use_dsa: whether to use DSA for context tokens,
  //                      if not, only need to store K into the KV cache of indexer
  // Outputs:
  //   - topk_indices: top-k indices [batch, seq_len, n_heads, index_topk]
  Status Forward(const Tensor& x, const Tensor& qr, const Tensor& workspace, const bool context_use_dsa,
                 Tensor& topk_indices, ForwardingContext& forwarding_context);

 private:
  const int layer_idx_;

  // Model parameters (actively used)
  size_t n_heads_;     // number of index heads
  size_t head_dim_;    // index head dimension
  size_t index_topk_;  // top-k value
  size_t block_size_;  // block size for quantization

  // Layers
  std::shared_ptr<Linear> wq_b_;          // Query projection (q_lora_rank -> n_heads * head_dim)
  std::shared_ptr<Linear> wk_;            // Key projection (dim -> head_dim)
  std::shared_ptr<Linear> weights_proj_;  // Weights projection (dim -> n_heads)

  // Key layer norm parameters
  Tensor k_norm_weight_;
  Tensor k_norm_bias_;
  float k_norm_eps_;

  // Flash and paged indexer layers
  std::unique_ptr<BaseLayer> flash_sparse_mla_indexer_layer_;
  std::unique_ptr<BaseLayer> paged_sparse_mla_indexer_layer_;
};

}  // namespace ksana_llm
