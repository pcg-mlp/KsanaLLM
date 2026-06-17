/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta_creator.h"

#include "executor/inflight_resource.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_compressor_slot_resource.h"

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
  meta->c4_indexer.Init(model_config, runtime_config, config_.rank);

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

AttentionMetaCreator::MetaArrays DeepSeekV4AttentionMetaCreator::BuildAttentionMetas(
    const std::vector<ForwardRequest>& forward_reqs, uint32_t attn_dp_group_id, SharedAttentionMeta& shared_meta) {
  // Reset MQA shared tensor shapes; must happen before the builders inspect them.
  auto& dsv4_shared = static_cast<DeepSeekV4SharedAttentionMeta&>(shared_meta);
  dsv4_shared.mqa.raw_out_loc_tensor.shape = {0};
  dsv4_shared.mqa.swa_block_indices_tensor.shape = {0};
  dsv4_shared.mqa.c128_block_indices_tensor.shape = {0};
  dsv4_shared.mqa.c128_topk_lengths_tensor.shape = {0};
  dsv4_shared.mqa.c4_num_splits_tensor.shape = {0};
  dsv4_shared.mqa.c128_num_splits_tensor.shape = {0};

  // Reset C4 indexer shared tensor shapes and statistics.
  dsv4_shared.c4_indexer.cu_seq_len_start_tensor.shape = {0};
  dsv4_shared.c4_indexer.cu_seq_len_end_tensor.shape = {0};
  dsv4_shared.c4_indexer.batch_indices_tensor.shape = {0};
  if (!dsv4_shared.c4_indexer.expanded_seq_len_tensor.shape.empty()) {
    dsv4_shared.c4_indexer.expanded_seq_len_tensor.shape = {0};
  }
  dsv4_shared.c4_indexer.c4_seq_lens_tensor.shape = {0};
  if (!dsv4_shared.c4_indexer.expanded_c4_seq_lens_tensor.shape.empty()) {
    dsv4_shared.c4_indexer.expanded_c4_seq_lens_tensor.shape = {0};
  }
  dsv4_shared.c4_indexer.dp_context_tokens = 0;
  dsv4_shared.c4_indexer.dp_decode_tokens = 0;
  dsv4_shared.c4_indexer.dp_context_kv_tokens = 0;

  // Use standard routing (ForwardRequestType::kFlash / kPage) for flash_meta / page_metas.
  // C4IndexerMeta, MQA and other modules all depend on page_metas following standard bucketing.
  // Compressor has its own internal IsPaged policy (q_seq_len==1) handled by PrepareCompressor below.
  MetaArrays result = AttentionMetaCreator::BuildAttentionMetas(forward_reqs, attn_dp_group_id, shared_meta);

  // Prepare compressor metadata. PrepareCompressor independently applies its own IsPaged routing
  // (q_seq_len==1 && kv_cached → paged compressor, others → flash compressor), which may differ
  // from the standard kFlash/kPage bucketing above.
  auto& dsv4_flash = static_cast<DeepSeekV4FlashAttentionMeta&>(*result.flash_meta);

  // Find the q_seq_len==1 paged bucket for the paged compressor path.
  DeepSeekV4PagedAttentionMeta* dsv4_paged = nullptr;
  for (auto& pi : result.page_metas) {
    if (pi->q_seq_len == 1) {
      dsv4_paged = static_cast<DeepSeekV4PagedAttentionMeta*>(pi.get());
      break;
    }
  }

  if (dsv4_paged) {
    PrepareCompressor(forward_reqs, dsv4_shared.compressor, dsv4_flash.compressor, dsv4_paged->compressor,
                      config_.context);
  } else {
    // Pure prefill batch: no q_seq_len==1 decode requests; sink paged output into a temporary.
    DeepSeekV4PagedAttentionMeta tmp_paged;
    PrepareCompressor(forward_reqs, dsv4_shared.compressor, dsv4_flash.compressor, tmp_paged.compressor,
                      config_.context);
  }

  return result;
}

std::unique_ptr<FlashAttentionMetaBuilder> DeepSeekV4AttentionMetaCreator::CreateFlashBuilder(
    const AttentionBuilderConfig& config) {
  return std::make_unique<DeepSeekV4FlashAttentionMetaBuilder>(config);
}

std::unique_ptr<PagedAttentionMetaBuilder> DeepSeekV4AttentionMetaCreator::CreatePagedBuilder(
    const AttentionBuilderConfig& config) {
  return std::make_unique<DeepSeekV4PagedAttentionMetaBuilder>(config);
}

void DeepSeekV4AttentionMetaCreator::WireInflightResources(InflightResourceManager* mgr,
                                                           SharedAttentionMeta* shared_meta) {
  if (!mgr || !shared_meta) {
    return;
  }
  // Creator guarantees shared_meta is DeepSeekV4SharedAttentionMeta (created by
  // CreateSharedAttentionMeta above), so static_cast is safe here.
  auto* dsv4 = static_cast<DeepSeekV4SharedAttentionMeta*>(shared_meta);
  dsv4->compressor.slot_resource_ = mgr->GetTypedResource<CompressorSlotResource>("compressor_slot");
}

}  // namespace ksana_llm
