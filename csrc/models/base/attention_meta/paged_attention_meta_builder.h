/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/attention_meta/attention_meta_builder.h"

namespace ksana_llm {

// Abstract base for paged (decode) attention input builders.
class PagedAttentionMetaBuilder {
 public:
  virtual ~PagedAttentionMetaBuilder() = default;
  virtual void Build(PagedAttentionMeta& meta, SharedAttentionMeta& shared_meta) = 0;
};

// Default paged builder: owns the paged-specific prepare logic.
class StandardPagedAttentionMetaBuilder : public PagedAttentionMetaBuilder {
 public:
  explicit StandardPagedAttentionMetaBuilder(const AttentionBuilderConfig& config);

  // Prepares all paged-path tensors in `meta`.
  void Build(PagedAttentionMeta& meta, SharedAttentionMeta& shared_meta) override;

 protected:
  AttentionBuilderConfig config_;

  // Template-method hook — override to customise layer_kv_cache_ptr capacity and block_table setup.
  // Default: standard capacity (layer_num_on_node × 2).
  virtual void OnPrepareKVCacheBlockTable(StandardAttentionMeta& meta, StandardSharedAttentionMeta& shared_meta);

 private:
  // Prepare input_length tensor.
  void PrepareInputLength(PagedAttentionMeta& meta, SharedAttentionMeta& shared_meta);

  // Prepare rotary embedding position and mask tensors for the decode (paged) path.
  void PrepareDecodeRotary(PagedAttentionMeta& meta, SharedAttentionMeta& shared_meta);
};

}  // namespace ksana_llm
