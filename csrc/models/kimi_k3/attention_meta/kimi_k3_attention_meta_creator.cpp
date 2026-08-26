/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/kimi_k3/attention_meta/kimi_k3_attention_meta_creator.h"

#include "executor/inflight_resource.h"

#ifdef ENABLE_CUDA
#  include "kernels/nvidia/kda_kernel_wrapper.h"
#else
constexpr int kKdaChunkSize = 64;
#endif

#include "models/deepseek_v4/attention_meta/deepseek_v4_compressor_slot_resource.h"

namespace ksana_llm {

std::unique_ptr<SharedAttentionMeta> KimiK3AttentionMetaCreator::CreateSharedAttentionMeta() {
  auto meta = std::make_unique<KimiK3SharedAttentionMeta>();
  InitDeepSeekV3SharedMeta(meta.get());

  // Initialize KimiK3-specific shared tensors.
  meta->Init(config_.model_config, config_.runtime_config, config_.layer_num_on_node, config_.rank,
             config_.block_cache_layout);

  return meta;
}

std::unique_ptr<FlashAttentionMeta> KimiK3AttentionMetaCreator::CreateFlashAttentionMeta() {
  auto meta = std::make_unique<KimiK3FlashAttentionMeta>();
  const size_t max_batch_size = config_.runtime_config.max_batch_size;
  meta->has_initial_state_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BOOL, {max_batch_size});

  // KDA chunked-prefill metadata. Worst-case chunk count = ceil(max_seq_len/BT) * max_batch_size.
  const size_t max_seq_len = config_.runtime_config.max_seq_len;
  const size_t max_num_chunks = ((max_seq_len + kKdaChunkSize - 1) / kKdaChunkSize) * max_batch_size;
  meta->chunk_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_num_chunks, 2});
  meta->chunk_offsets_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size + 1});
  return meta;
}

std::unique_ptr<PagedAttentionMeta> KimiK3AttentionMetaCreator::CreatePagedAttentionMeta() {
  return std::make_unique<KimiK3PagedAttentionMeta>();
}

AttentionMetaCreator::MetaArrays KimiK3AttentionMetaCreator::BuildAttentionMetas(
    const std::vector<ForwardRequest>& forward_reqs, uint32_t attn_dp_group_id, SharedAttentionMeta& shared_meta) {
  auto& kimi_shared = static_cast<KimiK3SharedAttentionMeta&>(shared_meta);
  kimi_shared.state_indices_tensor.shape = {0};

  return DeepSeekV3AttentionMetaCreator::BuildAttentionMetas(forward_reqs, attn_dp_group_id, shared_meta);
}

std::unique_ptr<FlashAttentionMetaBuilder> KimiK3AttentionMetaCreator::CreateFlashBuilder(
    const AttentionBuilderConfig& config) {
  return std::make_unique<KimiK3FlashAttentionMetaBuilder>(config);
}

std::unique_ptr<PagedAttentionMetaBuilder> KimiK3AttentionMetaCreator::CreatePagedBuilder(
    const AttentionBuilderConfig& config) {
  return std::make_unique<KimiK3PagedAttentionMetaBuilder>(config);
}

void KimiK3AttentionMetaCreator::WireInflightResources(InflightResourceManager* mgr, SharedAttentionMeta* shared_meta) {
  if (!mgr || !shared_meta) {
    return;
  }
  auto* kimi = static_cast<KimiK3SharedAttentionMeta*>(shared_meta);
  kimi->slot_resource_ = mgr->GetTypedResource<CompressorSlotResource>("mamba_slot");
}

}  // namespace ksana_llm
