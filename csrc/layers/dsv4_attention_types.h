/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "utils/tensor.h"

namespace ksana_llm {

// Typed input for FlashDsv4AttentionLayer and PagedDsv4AttentionLayer.
struct Dsv4AttentionInputs {
  const Tensor* rotary_embedding_pos;
  const Tensor* q;
  const Tensor* kv;
  const Tensor* layer_swa_kv_cache;
  const Tensor* tile_scheduler_metadata;
  const Tensor* num_splits;
  const Tensor* attn_sink;
  const Tensor* raw_out_loc;
  const Tensor* swa_indices;
  // Optional: non-null when compress_ratio > 0
  const Tensor* layer_extra_kv_cache = nullptr;
  const Tensor* extra_indices = nullptr;
  // Optional: non-null for c128 (compress_ratio == 128)
  const Tensor* extra_topk_length = nullptr;
};

}  // namespace ksana_llm
