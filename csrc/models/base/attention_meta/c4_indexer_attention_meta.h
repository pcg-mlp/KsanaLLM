/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "models/base/attention_meta/indexer_attention_meta.h"

namespace ksana_llm {

// Derived per-batch attention meta for the C4 Indexer component (DSV4 / compress_ratio=4).
// Extends IndexerAttentionMeta with C4-specific tensor views and host staging.
struct C4IndexerAttentionMeta : public IndexerAttentionMeta {
  // ── C4-specific tensor views (sliced from C4IndexerSharedAttentionMeta) ──
  Tensor c4_seq_lens_tensor;           // view
  Tensor extend_lens_tensor;           // view
  Tensor expanded_c4_seq_lens_tensor;  // view

  // ── C4-specific host staging vectors ─────────────────────────────────────
  std::vector<int> c4_seq_lens_host;
  std::vector<int> extend_lens_host;
  std::vector<int> expanded_c4_seq_lens_host;

  size_t total_dp_compressed_input_ids_len = 0;

  void Reset() override {
    IndexerAttentionMeta::Reset();
    c4_seq_lens_host.clear();
    extend_lens_host.clear();
    expanded_c4_seq_lens_host.clear();
    total_dp_compressed_input_ids_len = 0;
  }
};

}  // namespace ksana_llm
