/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "models/base/attention_meta/attention_meta_creator.h"
#include "models/deepseek_v3/attention_meta/deepseek_v3_attention_meta.h"
#include "models/deepseek_v3/attention_meta/deepseek_v3_attention_meta_builder.h"

namespace ksana_llm {

// Model-specific AttentionMetaCreator for DeepSeekV3.
// Responsibility:
//   - Create DeepSeekV3-typed FlashAttentionMeta / PagedAttentionMeta objects.
//   - Initialize MLA and DSA shared tensors in CreateSharedAttentionMeta(), keeping this
//     logic out of the base-class AttentionMetaCreator (framework layer).
class DeepSeekV3AttentionMetaCreator : public AttentionMetaCreator {
 public:
  explicit DeepSeekV3AttentionMetaCreator(const Config& config) : AttentionMetaCreator(config) {}

  // Creates a SharedAttentionMeta with standard tensors (via base class) plus
  // DeepSeekV3-specific MLA / DSA tensor allocations.
  std::unique_ptr<SharedAttentionMeta> CreateSharedAttentionMeta() override;

  std::unique_ptr<FlashAttentionMeta> CreateFlashAttentionMeta() override;

  std::unique_ptr<PagedAttentionMeta> CreatePagedAttentionMeta() override;

  // Returns DeepSeekV3-specific builders that handle MLA / DSA prepare steps.
  std::unique_ptr<FlashAttentionMetaBuilder> CreateFlashBuilder(const AttentionBuilderConfig& config) override;
  std::unique_ptr<PagedAttentionMetaBuilder> CreatePagedBuilder(const AttentionBuilderConfig& config) override;
};

}  // namespace ksana_llm
