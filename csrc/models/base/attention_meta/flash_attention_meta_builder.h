/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/attention_meta/attention_meta_builder.h"

namespace ksana_llm {

// Abstract base for flash (prefill) attention input builders.
// dp_total_prefix_len: the per-batch prefix length already written to ModelInput::dp_total_prefix_len
// by ParseFromRequests; passed explicitly here so future non-ModelInput-based implementations
// can receive it without reading it from ModelInput.
class FlashAttentionMetaBuilder {
 public:
  virtual ~FlashAttentionMetaBuilder() = default;
  virtual void Build(FlashAttentionMeta& meta, SharedAttentionMeta& shared_meta, size_t dp_total_prefix_len) = 0;
};

// Default flash builder: owns the flash-specific prepare logic.
class StandardFlashAttentionMetaBuilder : public FlashAttentionMetaBuilder {
 public:
  explicit StandardFlashAttentionMetaBuilder(const AttentionBuilderConfig& config);

  // Prepares all flash-path tensors in `meta`.
  void Build(FlashAttentionMeta& meta, SharedAttentionMeta& shared_meta, size_t dp_total_prefix_len) override;

 protected:
  AttentionBuilderConfig config_;

  // Template-method hook — override to customise layer_kv_cache_ptr capacity and block_table setup.
  // Default: standard capacity (layer_num_on_node × 2).
  virtual void OnPrepareKVCacheBlockTable(StandardAttentionMeta& meta, StandardSharedAttentionMeta& shared_meta);

 private:
  // Prepare flexible-cache copy tasks (block IDs + token indices) for the flash path.
  void PrepareFlexibleCache(FlashAttentionMeta& meta, SharedAttentionMeta& shared_meta);

  // Prepare rotary embedding position and mask tensors for the flash (prefill) path.
  void PrepareFlashRotary(FlashAttentionMeta& meta, SharedAttentionMeta& shared_meta, size_t dp_total_prefix_len);
};

}  // namespace ksana_llm
