/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/deepseek_v3/attention_meta/deepseek_v3_attention_meta_builder.h"
#include "models/kimi_k3/attention_meta/kimi_k3_attention_meta.h"

namespace ksana_llm {

// KimiK3 flash builder extends DeepSeekV3's flash builder.
class KimiK3FlashAttentionMetaBuilder : public DeepSeekV3FlashAttentionMetaBuilder {
 public:
  explicit KimiK3FlashAttentionMetaBuilder(const AttentionBuilderConfig& config)
      : DeepSeekV3FlashAttentionMetaBuilder(config) {}

  void Build(FlashAttentionMeta& meta, SharedAttentionMeta& shared_meta, size_t dp_total_prefix_len) override;

 private:
  void PrepareStateIndices(KimiK3FlashAttentionMeta& meta, KimiK3SharedAttentionMeta& kimi_shared);
  void PrepareConv1D(KimiK3FlashAttentionMeta& meta);
  void PrepareChunkKDA(KimiK3FlashAttentionMeta& meta);
};

// KimiK3 paged builder extends DeepSeekV3's paged builder.
class KimiK3PagedAttentionMetaBuilder : public DeepSeekV3PagedAttentionMetaBuilder {
 public:
  explicit KimiK3PagedAttentionMetaBuilder(const AttentionBuilderConfig& config)
      : DeepSeekV3PagedAttentionMetaBuilder(config) {}

  void Build(PagedAttentionMeta& meta, SharedAttentionMeta& shared_meta) override;

 private:
  void PrepareStateIndices(KimiK3PagedAttentionMeta& meta, KimiK3SharedAttentionMeta& kimi_shared);
};

}  // namespace ksana_llm
