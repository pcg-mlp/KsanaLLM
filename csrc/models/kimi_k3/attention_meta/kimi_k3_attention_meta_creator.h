/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "models/deepseek_v3/attention_meta/deepseek_v3_attention_meta_creator.h"
#include "models/kimi_k3/attention_meta/kimi_k3_attention_meta.h"
#include "models/kimi_k3/attention_meta/kimi_k3_attention_meta_builder.h"

namespace ksana_llm {

// KimiK3 AttentionMetaCreator extends DeepSeekV3's creator.
// Inherits all MLA / DSA / Indexer initialization; additionally handles KDA.
class KimiK3AttentionMetaCreator : public DeepSeekV3AttentionMetaCreator {
 public:
  explicit KimiK3AttentionMetaCreator(const Config& config) : DeepSeekV3AttentionMetaCreator(config) {}

  // Override to create KimiK3-typed objects with KDA fields.
  std::unique_ptr<SharedAttentionMeta> CreateSharedAttentionMeta() override;
  std::unique_ptr<FlashAttentionMeta> CreateFlashAttentionMeta() override;
  std::unique_ptr<PagedAttentionMeta> CreatePagedAttentionMeta() override;

  // Reset the shared tensor shape.
  MetaArrays BuildAttentionMetas(const std::vector<ForwardRequest>& forward_reqs, uint32_t attn_dp_group_id,
                                 SharedAttentionMeta& shared_meta) override;

  // Builders return KimiK3-specific builders that add KDA steps.
  std::unique_ptr<FlashAttentionMetaBuilder> CreateFlashBuilder(const AttentionBuilderConfig& config) override;
  std::unique_ptr<PagedAttentionMetaBuilder> CreatePagedBuilder(const AttentionBuilderConfig& config) override;

  void WireInflightResources(InflightResourceManager* mgr, SharedAttentionMeta* shared_meta) override;
};

}  // namespace ksana_llm
