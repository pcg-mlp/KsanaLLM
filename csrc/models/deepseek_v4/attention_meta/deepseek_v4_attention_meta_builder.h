/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/attention_meta/dsa_attention_meta_builder.h"
#include "models/base/attention_meta/flash_attention_meta_builder.h"
#include "models/base/attention_meta/mla_attention_meta_builder.h"
#include "models/base/attention_meta/paged_attention_meta_builder.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_mqa_builder.h"

namespace ksana_llm {

// Flash (prefill) attention input builder for DeepSeekV4.
// Adds MLA/DSA-specific steps on top of the standard flash builder.
class DeepSeekV4FlashAttentionMetaBuilder : public StandardFlashAttentionMetaBuilder {
 public:
  explicit DeepSeekV4FlashAttentionMetaBuilder(const AttentionBuilderConfig& config);

  void Build(FlashAttentionMeta& meta, SharedAttentionMeta& shared_meta, size_t dp_total_prefix_len) override;

 protected:
  void OnPrepareKVCacheBlockTable(StandardAttentionMeta& meta, StandardSharedAttentionMeta& shared_meta) override;
};

// Paged (decode) attention input builder for DeepSeekV4.
// Adds MLA/DSA-specific steps on top of the standard paged builder.
class DeepSeekV4PagedAttentionMetaBuilder : public StandardPagedAttentionMetaBuilder {
 public:
  explicit DeepSeekV4PagedAttentionMetaBuilder(const AttentionBuilderConfig& config);

  void Build(PagedAttentionMeta& meta, SharedAttentionMeta& shared_meta) override;

 protected:
  void OnPrepareKVCacheBlockTable(StandardAttentionMeta& meta, StandardSharedAttentionMeta& shared_meta) override;
};

}  // namespace ksana_llm
