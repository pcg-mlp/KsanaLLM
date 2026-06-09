/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/attention_meta/indexer_shared_attention_meta.h"

namespace ksana_llm {

// Derived shared attention meta for the C4 Indexer component (DSV4 / compress_ratio=4).
// Extends IndexerSharedAttentionMeta with C4-specific shared device tensors and statistics.
struct C4IndexerSharedAttentionMeta : public IndexerSharedAttentionMeta {
  // ── C4-specific shared device tensors ────────────────────────────────────
  Tensor c4_seq_lens_tensor;           // DEVICE, INT32, [max_batch_size]
  // Allocated only when GetDecodeTokenNumThreshold() > 1.
  Tensor expanded_c4_seq_lens_tensor;  // DEVICE, INT32, [max_batch_size, decode_token_num_threshold]
  // Interface placeholder: indexer kernel parses inputs positionally so a valid tensor is required.
  Tensor gathered_c4_out_loc_tensor;   // DEVICE, INT32, [1]

  // ── C4-specific running statistics ────────────────────────────────────────
  size_t dp_context_kv_tokens = 0;

  // Allocates base + C4-specific tensors for the C4 (compress_ratio=4) path.
  // No-op when model_config.use_dsv4_attn is false.
  void Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank) override;
};

}  // namespace ksana_llm
