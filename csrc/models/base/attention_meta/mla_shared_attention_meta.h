/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "models/base/attention_meta/attention_meta.h"

namespace ksana_llm {

// MLA-specific shared tensors (tile scheduler metadata, num_splits, chunked-KV buffers).
// All tensors are pre-allocated once at construction and written per-batch.
struct MLASharedAttentionMeta {
  // Tile-scheduler metadata pool and per-batch num_splits.
  std::vector<Tensor> tile_scheduler_metadatas;
  Tensor num_splits;

  // Chunked-KV buffers (flash/prefill path only).
  // Pre-allocated to the maximum possible sizes; shapes are overwritten per batch
  // by PrepareChunkKV() to reflect the actual number of chunks.
  Tensor chunk_kv_batch_starts_host;      // HOST, INT32, [max_batch_size + 1]
  Tensor chunk_kv_cu_extend_lens_tensor;  // DEVICE, INT32, [2 * max_batch_size]
  Tensor chunk_kv_cu_seq_lens_tensor;     // DEVICE, INT32, [2 * max_batch_size]

  // Intermediate host staging vectors that must outlive the async H2D memcpy.
  std::vector<int> chunk_kv_cu_extend_lens_list;
  std::vector<int> chunk_kv_cu_seq_lens_list;

  // Maximum total KV tokens allowed per chunk (0 means chunking is disabled).
  int max_chunked_kv_tokens = 0;

  // Allocates MLA tensors based on model/runtime config. No-op when use_mla is false.
  void Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config, size_t decode_token_num_threshold,
            int rank);
};

}  // namespace ksana_llm
