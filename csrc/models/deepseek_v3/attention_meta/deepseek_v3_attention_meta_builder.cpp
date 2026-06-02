/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v3/attention_meta/deepseek_v3_attention_meta_builder.h"
#include "models/base/attention_meta/indexer_attention_meta_builder.h"

namespace ksana_llm {

using namespace attention_meta_builder_detail;

// ── Flash builder ──────────────────────────────────────────────────────────

DeepSeekV3FlashAttentionMetaBuilder::DeepSeekV3FlashAttentionMetaBuilder(const AttentionBuilderConfig& config)
    : StandardFlashAttentionMetaBuilder(config) {}

void DeepSeekV3FlashAttentionMetaBuilder::OnPrepareKVCacheBlockTable(StandardAttentionMeta& meta,
                                                                     StandardSharedAttentionMeta& shared_meta) {
  const int layer_factor = config_.model_config.use_dsa ? 1 : config_.layer_num_on_node;
  const int kv_factor = config_.model_config.use_mla ? 1 : 2;
  PrepareKVCacheBlockTable(meta, shared_meta, config_, layer_factor, kv_factor);
}

bool DeepSeekV3FlashAttentionMetaBuilder::Build(FlashAttentionMeta& meta, SharedAttentionMeta& shared_meta,
                                                size_t dp_total_prefix_len) {
  // super::Build() calls OnPrepareKVCacheBlockTable() virtually → dispatches to our override.
  const bool use_cache = StandardFlashAttentionMetaBuilder::Build(meta, shared_meta, dp_total_prefix_len);

  auto& dsv3_meta = static_cast<DeepSeekV3FlashAttentionMeta&>(meta);
  auto& standard_meta = static_cast<StandardAttentionMeta&>(meta);

  // MLA+DSA prefill path: PrepareFlashMla is only needed when use_dsa is also true
  // (pure-MLA prefill does not require tile_scheduler_metadata).
  if (config_.model_config.use_mla && config_.model_config.use_dsa) {
    PrepareFlashMla(dsv3_meta.mla, dsv3_meta.dsa, *shared_meta.GetMLA(), *shared_meta.GetDSA(), standard_meta, config_);
  }
  if (use_cache && config_.model_config.use_dsa) {
    PrepareStoreOutLoc(dsv3_meta.dsa, *shared_meta.GetDSA(), meta, config_);
    PrepareDSAExpandedBlockTable(dsv3_meta.dsa, *shared_meta.GetDSA(), meta, config_);
  }
  // Chunked-KV offset tensors for the MLA flash path (pure-MLA and MLA+DSA).
  if (config_.model_config.use_mla && !meta.dp_reqs.empty()) {
    PrepareChunkKV(*shared_meta.GetMLA(), meta, config_);
  }
  if (config_.model_config.use_dsa && !meta.dp_reqs.empty()) {
    PrepareFlashIndexer(dsv3_meta.indexer, *shared_meta.GetIndexer(), meta, config_);
  }

  return use_cache;
}

// ── Paged builder ──────────────────────────────────────────────────────────

DeepSeekV3PagedAttentionMetaBuilder::DeepSeekV3PagedAttentionMetaBuilder(const AttentionBuilderConfig& config)
    : StandardPagedAttentionMetaBuilder(config) {}

void DeepSeekV3PagedAttentionMetaBuilder::OnPrepareKVCacheBlockTable(StandardAttentionMeta& meta,
                                                                     StandardSharedAttentionMeta& shared_meta) {
  const int layer_factor = config_.model_config.use_dsa ? 1 : config_.layer_num_on_node;
  const int kv_factor = config_.model_config.use_mla ? 1 : 2;
  PrepareKVCacheBlockTable(meta, shared_meta, config_, layer_factor, kv_factor);
}

void DeepSeekV3PagedAttentionMetaBuilder::Build(PagedAttentionMeta& meta, SharedAttentionMeta& shared_meta) {
  // super::Build() calls OnPrepareKVCacheBlockTable() virtually.
  StandardPagedAttentionMetaBuilder::Build(meta, shared_meta);

  auto& dsv3_meta = static_cast<DeepSeekV3PagedAttentionMeta&>(meta);
  auto& standard_meta = static_cast<StandardAttentionMeta&>(meta);

  if (config_.model_config.use_mla) {
    // Decode path always needs tile_scheduler_metadata regardless of use_dsa.
    PrepareFlashMla(dsv3_meta.mla, dsv3_meta.dsa, *shared_meta.GetMLA(), *shared_meta.GetDSA(), standard_meta, config_);
  }
  if (config_.model_config.use_dsa) {
    PrepareStoreOutLoc(dsv3_meta.dsa, *shared_meta.GetDSA(), meta, config_);
    PrepareDSAExpandedBlockTable(dsv3_meta.dsa, *shared_meta.GetDSA(), meta, config_);
  }
  if (config_.model_config.use_dsa && !meta.dp_reqs.empty()) {
    PreparePagedIndexer(dsv3_meta.indexer, *shared_meta.GetIndexer(), meta, config_);
  }
}

}  // namespace ksana_llm
