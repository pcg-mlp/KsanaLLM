/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "utils/tensor.h"

namespace ksana_llm {

// Base per-batch attention meta for the Indexer component (DSV3 / compress_ratio=1).
// Each FlashAttentionMeta / PagedAttentionMeta embeds one instance as the `indexer` member.
// All tensor fields are views sliced from IndexerSharedAttentionMeta backing tensors.
struct IndexerAttentionMeta {
  // ── Tensor views (sliced from IndexerSharedAttentionMeta) ────────────────
  Tensor cu_seq_len_start_tensor;  // view
  Tensor cu_seq_len_end_tensor;    // view
  Tensor batch_indices_tensor;     // view
  Tensor expanded_seq_len_tensor;  // view

  // ── Host staging vectors (must outlive async H2D memcpy) ─────────────────
  std::vector<int> cu_seq_len_start_host;
  std::vector<int> cu_seq_len_end_host;
  std::vector<int> batch_indices_host;
  std::vector<int> expanded_seq_len_host;

  size_t total_dp_input_ids_len = 0;

  virtual void Reset() {
    cu_seq_len_start_host.clear();
    cu_seq_len_end_host.clear();
    batch_indices_host.clear();
    expanded_seq_len_host.clear();
    total_dp_input_ids_len = 0;
  }

  virtual ~IndexerAttentionMeta() = default;
};

}  // namespace ksana_llm
