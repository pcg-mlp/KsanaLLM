/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/attention_meta/attention_meta.h"
#include "models/base/attention_meta/attention_meta_builder.h"
#include "models/base/attention_meta/dsa_shared_attention_meta.h"
#include "models/base/attention_meta/mla_shared_attention_meta.h"

namespace ksana_llm {

// Prepare per-batch MLA attention metadata: selects the pre-allocated tile_scheduler_metadata
// view and assigns a num_splits slice from the shared pool.
//
// When config.model_config.use_dsa is also true, handles the MLA+DSA interaction:
// marks tile_scheduler_metadata as uninitialized (filled by DSA kernel) and sets up
// paged_schedule_meta from the shared pool.
//
// Caller must ensure config.model_config.use_mla is true before calling.
// Only active under ENABLE_CUDA; no-op on other backends.
void PrepareFlashMla(MLAAttentionMeta& mla_meta, DSAAttentionMeta& dsa_meta, MLASharedAttentionMeta& mla_shared,
                     DSASharedAttentionMeta& dsa_shared, const StandardAttentionMeta& input,
                     const AttentionBuilderConfig& config);

// Prepare chunked-KV offset tensors for the MLA flash (prefill) path.
// Greedily packs consecutive requests in flash_meta.dp_reqs into chunks bounded by
// mla_shared.max_chunked_kv_tokens, then uploads the resulting cumulative-length
// arrays to device via async H2D memcpy.
// The shapes of chunk_kv_batch_starts_host, chunk_kv_cu_extend_lens_tensor, and
// chunk_kv_cu_seq_lens_tensor reflect the actual chunk count for this batch.
//
// Caller must ensure flash_meta.dp_reqs is non-empty.
// Active under ENABLE_CUDA and ENABLE_KUNLUN; no-op on other backends.
void PrepareChunkKV(MLASharedAttentionMeta& mla_shared, const FlashAttentionMeta& flash_meta,
                    const AttentionBuilderConfig& config);

}  // namespace ksana_llm
