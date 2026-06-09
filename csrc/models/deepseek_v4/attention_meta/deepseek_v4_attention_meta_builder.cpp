/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta_builder.h"

#include "models/base/attention_meta/c4_indexer_attention_meta_builder.h"

namespace ksana_llm {

using namespace attention_meta_builder_detail;

// ── Flash builder ──────────────────────────────────────────────────────────

DeepSeekV4FlashAttentionMetaBuilder::DeepSeekV4FlashAttentionMetaBuilder(const AttentionBuilderConfig& config)
    : StandardFlashAttentionMetaBuilder(config) {}

void DeepSeekV4FlashAttentionMetaBuilder::OnPrepareKVCacheBlockTable(StandardAttentionMeta& meta,
                                                                     StandardSharedAttentionMeta& shared_meta) {
  const int layer_factor = config_.model_config.use_dsa ? 1 : config_.layer_num_on_node;
  const int kv_factor = config_.model_config.use_mla ? 1 : 2;
  PrepareKVCacheBlockTable(meta, shared_meta, config_, layer_factor, kv_factor);
}

void DeepSeekV4FlashAttentionMetaBuilder::Build(FlashAttentionMeta& meta, SharedAttentionMeta& shared_meta,
                                                size_t dp_total_prefix_len) {
  StandardFlashAttentionMetaBuilder::Build(meta, shared_meta, dp_total_prefix_len);

  auto& dsv4_meta = static_cast<DeepSeekV4FlashAttentionMeta&>(meta);
  auto& standard_meta = static_cast<StandardAttentionMeta&>(meta);

  if (config_.model_config.use_mla && config_.model_config.use_dsa) {
    PrepareFlashMla(dsv4_meta.mla, dsv4_meta.dsa, *shared_meta.GetMLA(), *shared_meta.GetDSA(), standard_meta, config_);
  }
  if (meta.use_cache && config_.model_config.use_dsa) {
    PrepareStoreOutLoc(dsv4_meta.dsa, *shared_meta.GetDSA(), meta, config_);
    PrepareDSAExpandedBlockTable(dsv4_meta.dsa, *shared_meta.GetDSA(), meta, config_);
  }
  // Chunked-KV offset tensors for the MLA flash path (pure-MLA and MLA+DSA).
  if (config_.model_config.use_mla && !meta.dp_reqs.empty()) {
    PrepareChunkKV(*shared_meta.GetMLA(), meta, config_);
  }
  if (config_.model_config.use_dsv4_attn) {
    PrepareMqaFlash(dsv4_meta.mqa, *shared_meta.GetMqa(), meta, config_);
  }
  if (config_.model_config.use_dsv4_attn && !meta.dp_reqs.empty()) {
    auto& dsv4_shared = static_cast<DeepSeekV4SharedAttentionMeta&>(shared_meta);
    PrepareFlashC4Indexer(dsv4_meta.c4_indexer, dsv4_shared.c4_indexer, meta, config_);
  }
}

// ── Paged builder ──────────────────────────────────────────────────────────

DeepSeekV4PagedAttentionMetaBuilder::DeepSeekV4PagedAttentionMetaBuilder(const AttentionBuilderConfig& config)
    : StandardPagedAttentionMetaBuilder(config) {}

void DeepSeekV4PagedAttentionMetaBuilder::OnPrepareKVCacheBlockTable(StandardAttentionMeta& meta,
                                                                     StandardSharedAttentionMeta& shared_meta) {
  const int layer_factor = config_.model_config.use_dsa ? 1 : config_.layer_num_on_node;
  const int kv_factor = config_.model_config.use_mla ? 1 : 2;
  PrepareKVCacheBlockTable(meta, shared_meta, config_, layer_factor, kv_factor);
}

void DeepSeekV4PagedAttentionMetaBuilder::Build(PagedAttentionMeta& meta, SharedAttentionMeta& shared_meta) {
  StandardPagedAttentionMetaBuilder::Build(meta, shared_meta);

  auto& dsv4_meta = static_cast<DeepSeekV4PagedAttentionMeta&>(meta);
  auto& standard_meta = static_cast<StandardAttentionMeta&>(meta);

  if (config_.model_config.use_mla) {
    PrepareFlashMla(dsv4_meta.mla, dsv4_meta.dsa, *shared_meta.GetMLA(), *shared_meta.GetDSA(), standard_meta, config_);
  }
  if (config_.model_config.use_dsa) {
    PrepareStoreOutLoc(dsv4_meta.dsa, *shared_meta.GetDSA(), meta, config_);
    PrepareDSAExpandedBlockTable(dsv4_meta.dsa, *shared_meta.GetDSA(), meta, config_);
  }
  if (config_.model_config.use_dsv4_attn) {
    PrepareMqaPaged(dsv4_meta.mqa, *shared_meta.GetMqa(), meta, config_);
  }
  if (config_.model_config.use_dsv4_attn && !meta.dp_reqs.empty()) {
    auto& dsv4_shared = static_cast<DeepSeekV4SharedAttentionMeta&>(shared_meta);
    PreparePagedC4Indexer(dsv4_meta.c4_indexer, dsv4_shared.c4_indexer, meta, config_);
  }
}

}  // namespace ksana_llm
