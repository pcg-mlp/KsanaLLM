/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/attention_meta/attention_meta.h"
#include "models/base/attention_meta/attention_meta_builder.h"
#include "models/base/attention_meta/c4_indexer_attention_meta.h"
#include "models/base/attention_meta/c4_indexer_shared_attention_meta.h"
#include "models/base/attention_meta/indexer_attention_meta_builder.h"

namespace ksana_llm {

// Prepare flash (prefill) C4 indexer tensors for one batch.
// Calls PrepareFlashIndexer (base), then prepares C4-specific extend_lens.
// Adds compressed context KV tokens to indexer_shared.dp_context_kv_tokens.
void PrepareFlashC4Indexer(C4IndexerAttentionMeta& indexer_meta, C4IndexerSharedAttentionMeta& indexer_shared,
                            const FlashAttentionMeta& flash_meta, const AttentionBuilderConfig& config);

// Prepare paged (decode) C4 indexer tensors for one batch.
// Calls PreparePagedIndexer (base), then prepares C4-specific c4_seq_lens and expanded_c4_seq_lens.
void PreparePagedC4Indexer(C4IndexerAttentionMeta& indexer_meta, C4IndexerSharedAttentionMeta& indexer_shared,
                            const PagedAttentionMeta& page_meta, const AttentionBuilderConfig& config);

}  // namespace ksana_llm
