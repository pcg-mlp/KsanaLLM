/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v3/attention_meta/deepseek_v3_attention_meta_creator.h"

namespace ksana_llm {

void DeepSeekV3AttentionMetaCreator::InitDeepSeekV3SharedMeta(DeepSeekV3SharedAttentionMeta* meta) {
  InitStandardSharedAttentionMeta(*meta);

  const ModelConfig& model_config = config_.model_config;
  const RuntimeConfig& runtime_config = config_.runtime_config;

  meta->dsa.Init(model_config, runtime_config, config_.layer_num_on_node, config_.max_table_block_num,
                 config_.max_token_num, config_.decode_token_num_threshold);
  meta->mla.Init(model_config, runtime_config, config_.decode_token_num_threshold, config_.rank);
  meta->indexer.Init(model_config, runtime_config, config_.rank);

  if (runtime_config.enable_flexible_caching && model_config.use_mla) {
    meta->dp_dst_flexible_rotary_embedding_pos =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {config_.max_token_num});
  }

  if (!runtime_config.enable_flexible_caching && config_.enable_blocked_multi_token_forwarding_kv &&
      model_config.use_mla) {
    meta->dp_flexible_offset_uint64_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {runtime_config.max_batch_size + 1});
  }
}

std::unique_ptr<SharedAttentionMeta> DeepSeekV3AttentionMetaCreator::CreateSharedAttentionMeta() {
  auto meta = std::make_unique<DeepSeekV3SharedAttentionMeta>();
  InitDeepSeekV3SharedMeta(meta.get());
  return meta;
}

std::unique_ptr<FlashAttentionMeta> DeepSeekV3AttentionMetaCreator::CreateFlashAttentionMeta() {
  return std::make_unique<DeepSeekV3FlashAttentionMeta>();
}

std::unique_ptr<PagedAttentionMeta> DeepSeekV3AttentionMetaCreator::CreatePagedAttentionMeta() {
  return std::make_unique<DeepSeekV3PagedAttentionMeta>();
}

std::unique_ptr<FlashAttentionMetaBuilder> DeepSeekV3AttentionMetaCreator::CreateFlashBuilder(
    const AttentionBuilderConfig& config) {
  return std::make_unique<DeepSeekV3FlashAttentionMetaBuilder>(config);
}

std::unique_ptr<PagedAttentionMetaBuilder> DeepSeekV3AttentionMetaCreator::CreatePagedBuilder(
    const AttentionBuilderConfig& config) {
  return std::make_unique<DeepSeekV3PagedAttentionMetaBuilder>(config);
}

AttentionMetaCreator::MetaArrays DeepSeekV3AttentionMetaCreator::BuildAttentionMetas(
    const std::vector<ForwardRequest>& forward_reqs, uint32_t attn_dp_group_id,
    SharedAttentionMeta& shared_meta) {
  // Reset indexer shared tensor cursors and statistics before each batch.
  auto& dsv3_shared = static_cast<DeepSeekV3SharedAttentionMeta&>(shared_meta);
  dsv3_shared.indexer.cu_seq_len_start_tensor.shape = {0};
  dsv3_shared.indexer.cu_seq_len_end_tensor.shape = {0};
  dsv3_shared.indexer.batch_indices_tensor.shape = {0};
  if (!dsv3_shared.indexer.expanded_seq_len_tensor.shape.empty()) {
    dsv3_shared.indexer.expanded_seq_len_tensor.shape = {0};
  }
  dsv3_shared.indexer.dp_context_tokens = 0;
  dsv3_shared.indexer.dp_decode_tokens = 0;

  return AttentionMetaCreator::BuildAttentionMetas(forward_reqs, attn_dp_group_id, shared_meta);
}

}  // namespace ksana_llm
