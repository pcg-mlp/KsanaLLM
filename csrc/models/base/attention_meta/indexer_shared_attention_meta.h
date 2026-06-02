/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>

#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "utils/tensor.h"

namespace ksana_llm {

// Base shared attention meta for the Indexer component (DSV3 / compress_ratio=1).
// Holds shared device backing tensors (per-batch views are sliced from these) and
// persistent host staging buffers. Mirrors the shared-tensor half of IndexerMeta.
struct IndexerSharedAttentionMeta {
  // ── Shared device backing tensors ─────────────────────────────────────────
  Tensor cu_seq_len_start_tensor;  // DEVICE, INT32, [max_token_num]
  Tensor cu_seq_len_end_tensor;    // DEVICE, INT32, [max_token_num]
  Tensor batch_indices_tensor;     // DEVICE, INT32, [max_token_num]
  // Allocated only when GetDecodeTokenNumThreshold() > 1.
  Tensor expanded_seq_len_tensor;  // DEVICE, INT32, [max_batch_size, decode_token_num_threshold]

  // ── Host staging (cross-batch persistent) ─────────────────────────────────
  Tensor chunk_token_starts_host;     // HOST, INT32, [max_batch_size + 1]
  Tensor chunk_kv_token_ranges_host;  // HOST, INT32, [max_batch_size * 2]

  // ── Configuration (set by Init, read-only during Prepare*) ────────────────
  int compress_ratio = 1;
  int max_chunked_q_tokens = 0;
  int max_chunked_kv_tokens = 0;
  uint32_t attn_dp_group_id = 0;

  // ── Running statistics (written by PrepareFlashIndexer/PreparePagedIndexer) ──
  size_t dp_context_tokens = 0;
  size_t dp_decode_tokens = 0;

  // Allocates all tensors for the DSA (compress_ratio=1) path.
  // No-op when model_config.use_dsa is false.
  virtual void Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank);

  virtual ~IndexerSharedAttentionMeta() = default;

 protected:
  // Core allocation without the use_dsa guard; also called by derived class Init().
  // compress_ratio_val: 1 for DSV3 (base), 4 for DSV4 (C4 derived).
  void InitCore(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                int compress_ratio_val);
};

}  // namespace ksana_llm
