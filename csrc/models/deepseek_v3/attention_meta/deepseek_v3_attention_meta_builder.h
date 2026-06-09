/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/attention_meta/dsa_attention_meta_builder.h"
#include "models/base/attention_meta/flash_attention_meta_builder.h"
#include "models/base/attention_meta/mla_attention_meta_builder.h"
#include "models/base/attention_meta/paged_attention_meta_builder.h"
#include "models/deepseek_v3/attention_meta/deepseek_v3_attention_meta.h"

namespace ksana_llm {

// Flash (prefill) attention input builder for DeepSeekV3.
// Adds MLA/DSA-specific steps on top of the standard flash builder.
class DeepSeekV3FlashAttentionMetaBuilder : public StandardFlashAttentionMetaBuilder {
 public:
  explicit DeepSeekV3FlashAttentionMetaBuilder(const AttentionBuilderConfig& config);

  void Build(FlashAttentionMeta& meta, SharedAttentionMeta& shared_meta, size_t dp_total_prefix_len) override;

 protected:
  // Use DSA/MLA-specific capacity factors for layer_kv_cache_ptr initialisation.
  void OnPrepareKVCacheBlockTable(StandardAttentionMeta& meta, StandardSharedAttentionMeta& shared_meta) override;
};

// Paged (decode) attention input builder for DeepSeekV3.
// Adds MLA/DSA-specific steps on top of the standard paged builder.
class DeepSeekV3PagedAttentionMetaBuilder : public StandardPagedAttentionMetaBuilder {
 public:
  explicit DeepSeekV3PagedAttentionMetaBuilder(const AttentionBuilderConfig& config);

  void Build(PagedAttentionMeta& meta, SharedAttentionMeta& shared_meta) override;

 protected:
  // Use DSA/MLA-specific capacity factors for layer_kv_cache_ptr initialisation.
  void OnPrepareKVCacheBlockTable(StandardAttentionMeta& meta, StandardSharedAttentionMeta& shared_meta) override;
};

}  // namespace ksana_llm
