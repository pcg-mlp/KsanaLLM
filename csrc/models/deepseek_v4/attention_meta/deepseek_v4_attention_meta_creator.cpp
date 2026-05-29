/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta_creator.h"

namespace ksana_llm {

std::unique_ptr<SharedAttentionMeta> DeepSeekV4AttentionMetaCreator::CreateSharedAttentionMeta() {
  auto meta = std::make_unique<DeepSeekV4SharedAttentionMeta>();
  InitStandardSharedAttentionMeta(*meta);

  const ModelConfig& model_config = config_.model_config;
  const RuntimeConfig& runtime_config = config_.runtime_config;

  meta->dsa.Init(model_config, runtime_config, config_.layer_num_on_node, config_.max_table_block_num,
                 config_.max_token_num, config_.decode_token_num_threshold);
  meta->mla.Init(model_config, runtime_config, config_.decode_token_num_threshold, config_.rank);
  meta->compressor.Init(model_config, runtime_config, config_.layer_num_on_node, config_.layer_start_on_node,
                        config_.rank, config_.block_cache_layout);
  meta->mqa.Init(model_config, runtime_config, config_.layer_num_on_node, config_.rank, config_.block_cache_layout);

  if (runtime_config.enable_flexible_caching && model_config.use_mla) {
    meta->dp_dst_flexible_rotary_embedding_pos =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {config_.max_token_num});
  }

  if (!runtime_config.enable_flexible_caching && config_.enable_blocked_multi_token_forwarding_kv &&
      model_config.use_mla) {
    meta->dp_flexible_offset_uint64_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {runtime_config.max_batch_size + 1});
  }

  return meta;
}

std::unique_ptr<FlashAttentionMeta> DeepSeekV4AttentionMetaCreator::CreateFlashAttentionMeta() {
  return std::make_unique<DeepSeekV4FlashAttentionMeta>();
}

std::unique_ptr<PagedAttentionMeta> DeepSeekV4AttentionMetaCreator::CreatePagedAttentionMeta() {
  return std::make_unique<DeepSeekV4PagedAttentionMeta>();
}

std::unique_ptr<FlashAttentionMetaBuilder> DeepSeekV4AttentionMetaCreator::CreateFlashBuilder(
    const AttentionBuilderConfig& config) {
  return std::make_unique<DeepSeekV4FlashAttentionMetaBuilder>(config);
}

std::unique_ptr<PagedAttentionMetaBuilder> DeepSeekV4AttentionMetaCreator::CreatePagedBuilder(
    const AttentionBuilderConfig& config) {
  return std::make_unique<DeepSeekV4PagedAttentionMetaBuilder>(config);
}

}  // namespace ksana_llm
