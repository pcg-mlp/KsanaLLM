/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/attention_meta/attention_meta.h"
#include "models/base/attention_meta/attention_meta_builder.h"
#include "models/base/attention_meta/indexer_attention_meta.h"
#include "models/base/attention_meta/indexer_shared_attention_meta.h"

namespace ksana_llm {

// Prepare flash (prefill) indexer tensors for one batch.
// Runs PrepareChunk, PrepareCuSeqLen (flash), and PrepareBatchIndices.
// Adds flash_meta.total_dp_input_ids_len to indexer_shared.dp_context_tokens.
void PrepareFlashIndexer(IndexerAttentionMeta& indexer_meta, IndexerSharedAttentionMeta& indexer_shared,
                         const FlashAttentionMeta& flash_meta, const AttentionBuilderConfig& config);

// Prepare paged (decode) indexer tensors for one batch.
// Runs PrepareCuSeqLen (paged) and PrepareSeqLen when q_seq_len > 1.
// Adds page_meta.total_dp_input_ids_len to indexer_shared.dp_decode_tokens.
void PreparePagedIndexer(IndexerAttentionMeta& indexer_meta, IndexerSharedAttentionMeta& indexer_shared,
                         const PagedAttentionMeta& page_meta, const AttentionBuilderConfig& config);

}  // namespace ksana_llm
