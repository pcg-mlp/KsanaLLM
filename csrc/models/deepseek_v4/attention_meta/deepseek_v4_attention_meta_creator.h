/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "models/base/attention_meta/attention_meta_creator.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta_builder.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_compressor_builder.h"

namespace ksana_llm {

// Model-specific AttentionMetaCreator for DeepSeekV4.
// Responsibility:
//   - Create DeepSeekV4-typed FlashAttentionMeta / PagedAttentionMeta objects.
//   - Initialize MLA, DSA and Compressor shared tensors in CreateSharedAttentionMeta(),
//     keeping this logic out of the base-class AttentionMetaCreator (framework layer).
class DeepSeekV4AttentionMetaCreator : public AttentionMetaCreator {
 public:
  explicit DeepSeekV4AttentionMetaCreator(const Config& config) : AttentionMetaCreator(config) {}

  // Creates a SharedAttentionMeta with standard tensors (via base class) plus
  // DeepSeekV4-specific MLA / DSA tensor allocations.
  std::unique_ptr<SharedAttentionMeta> CreateSharedAttentionMeta() override;

  std::unique_ptr<FlashAttentionMeta> CreateFlashAttentionMeta() override;

  std::unique_ptr<PagedAttentionMeta> CreatePagedAttentionMeta() override;

  // Uses Compressor routing: q_seq_len==1 (decode) → paged, all others → flash.
  // Also calls PrepareCompressor to populate CompressorAttentionMeta fields.
  MetaArrays BuildAttentionMetas(const std::vector<ForwardRequest>& forward_reqs, uint32_t attn_dp_group_id,
                                  SharedAttentionMeta& shared_meta) override;

  // Returns DeepSeekV4-specific builders that handle MLA / DSA prepare steps.
  std::unique_ptr<FlashAttentionMetaBuilder> CreateFlashBuilder(const AttentionBuilderConfig& config) override;
  std::unique_ptr<PagedAttentionMetaBuilder> CreatePagedBuilder(const AttentionBuilderConfig& config) override;
};

}  // namespace ksana_llm
