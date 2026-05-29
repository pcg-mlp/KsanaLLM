/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/attention_meta/attention_meta_creator.h"

#include <vector>

#include "models/base/attention_meta/flash_attention_meta_builder.h"
#include "models/base/attention_meta/paged_attention_meta_builder.h"

namespace ksana_llm {

void AttentionMetaCreator::InitStandardSharedAttentionMeta(StandardSharedAttentionMeta& shared_meta) {
  const RuntimeConfig& runtime_config = config_.runtime_config;
  const int layer_num_on_node = config_.layer_num_on_node;
  const size_t max_table_block_num = config_.max_table_block_num;
  const size_t max_token_num = config_.max_token_num;
  const size_t decode_token_num_threshold = config_.decode_token_num_threshold;
  const size_t max_batch_size = runtime_config.max_batch_size;

  shared_meta.input_length = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  shared_meta.kv_list = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_POINTER,
                               {static_cast<size_t>(layer_num_on_node), max_table_block_num, 2});
  shared_meta.layer_kv_cache_ptr =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {1 + static_cast<size_t>(layer_num_on_node * 2)});
  shared_meta.kv_cache_offset =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size + 1 + decode_token_num_threshold});
  shared_meta.rotary_embedding_pos = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_token_num});
  shared_meta.rotary_embedding_mask = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_token_num});
  shared_meta.block_table = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size * max_table_block_num});

  if (runtime_config.enable_flexible_caching) {
    shared_meta.dp_flexible_rotary_embedding_mask =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_token_num});
    shared_meta.dp_src_flexible_rotary_embedding_pos =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_token_num});
    shared_meta.dp_flexible_offset_uint64_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1});

    shared_meta.dp_dst_flexible_kv_cache_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_POINTER, {max_token_num * max_batch_size});
    shared_meta.dp_src_flexible_kv_cache_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_POINTER, {max_token_num * max_batch_size});
    shared_meta.dp_dst_flexible_token_idx_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num * max_batch_size});
    shared_meta.dp_src_flexible_token_idx_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num * max_batch_size});
  } else if (!config_.enable_blocked_multi_token_forwarding_kv) {
    shared_meta.dp_flexible_offset_uint64_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1});
  }
}

std::unique_ptr<SharedAttentionMeta> AttentionMetaCreator::CreateSharedAttentionMeta() {
  auto shared_meta = std::make_unique<StandardSharedAttentionMeta>();
  InitStandardSharedAttentionMeta(*shared_meta);
  return shared_meta;
}

std::unique_ptr<FlashAttentionMeta> AttentionMetaCreator::CreateFlashAttentionMeta() {
  return std::make_unique<FlashAttentionMeta>();
}

std::unique_ptr<PagedAttentionMeta> AttentionMetaCreator::CreatePagedAttentionMeta() {
  return std::make_unique<PagedAttentionMeta>();
}

std::unique_ptr<FlashAttentionMetaBuilder> AttentionMetaCreator::CreateFlashBuilder(
    const AttentionBuilderConfig& config) {
  return std::make_unique<StandardFlashAttentionMetaBuilder>(config);
}

std::unique_ptr<PagedAttentionMetaBuilder> AttentionMetaCreator::CreatePagedBuilder(
    const AttentionBuilderConfig& config) {
  return std::make_unique<StandardPagedAttentionMetaBuilder>(config);
}

}  // namespace ksana_llm
