/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/attention_meta/attention_meta_builder.h"

#include <algorithm>
#include <memory>
#include <vector>

#ifdef ENABLE_CUDA
#  include "kernels/nvidia/kernel_wrapper.h"
#endif
#include "configure/environment.h"
#include "profiler/profile_event.h"

namespace ksana_llm {
namespace attention_meta_builder_detail {

void PrepareKVCacheBlocks(StandardAttentionMeta& meta, StandardSharedAttentionMeta& shared_meta,
                          const AttentionBuilderConfig& config) {
  if (config.model_config.use_dsv4_attn) {  // dsv4 does not require non-contiguous KV lists
    return;
  }

  PROFILE_EVENT_SCOPE(PrepareKVCacheBlocks, "PrepareKVCacheBlocks", config.rank);
  const auto& reqs = meta.dp_reqs;

  std::vector<int> kv_cache_offset_host(reqs.size() + 1);
  kv_cache_offset_host[0] = 0;  // first is 0
  for (size_t i = 0; i < reqs.size(); ++i) {
    kv_cache_offset_host[i + 1] = reqs[i]->kv_cache_ptrs.size() + kv_cache_offset_host[i];
  }

  KLLM_LOG_DEBUG << "kv_cache_offset_host " << kv_cache_offset_host;

  meta.kv_cache_offset =
      shared_meta.kv_cache_offset.GetView({kv_cache_offset_host.size()}, shared_meta.kv_cache_offset.shape[0]);
  shared_meta.kv_cache_offset.shape[0] += meta.kv_cache_offset.GetElementNumber();
  MemcpyAsync(meta.kv_cache_offset.GetPtr<void>(), kv_cache_offset_host.data(),
              kv_cache_offset_host.size() * sizeof(decltype(kv_cache_offset_host)::value_type), MEMCPY_HOST_TO_DEVICE,
              config.context->device->GetD2HStream());

  const int total_block_num = kv_cache_offset_host.back();
  const bool host_fill_all_layers =
#ifdef ENABLE_CUDA
      false;
#else
      true;  // ASCEND already fills all layers on host.
#endif
  for (const auto& kv_cache_type : config.kv_cache_types) {
    if (kv_cache_type != "attention") {
      continue;
    }

    const size_t kv_list_layer_count =
#ifdef ENABLE_CUDA
        host_fill_all_layers ? static_cast<size_t>(config.layer_num_on_node) : 1;
#else
        static_cast<size_t>(config.layer_num_on_node);
#endif
    std::vector<void*> kv_list_host(kv_list_layer_count * total_block_num * 2);
    const auto& cached_layer_offsets = config.kv_cache_layer_offsets.at(kv_cache_type);
    for (size_t layer_i = 0; layer_i < kv_list_layer_count; ++layer_i) {
      const auto& layer_offsets = cached_layer_offsets[layer_i];
      void** k_ptr = kv_list_host.data() + layer_i * total_block_num * 2;
      void** v_ptr = kv_list_host.data() + layer_i * total_block_num * 2 + total_block_num;
      for (size_t req_i = 0; req_i < reqs.size(); ++req_i) {
        for (const auto& cache_ptr : reqs[req_i]->kv_cache_ptrs) {
          *k_ptr++ = cache_ptr + layer_offsets.k_offset;
          *v_ptr++ = cache_ptr + layer_offsets.v_offset;
        }
      }
    }

    Tensor* current_info_kv_list = &meta.kv_list;
    Tensor* current_kv_list = &shared_meta.kv_list;
    KLLM_LOG_DEBUG << kv_cache_type << " kv_list_host " << kv_list_host;
    *current_info_kv_list = current_kv_list->GetView(
        {static_cast<size_t>(config.layer_num_on_node), static_cast<size_t>(total_block_num * 2)},
        current_kv_list->shape[0]);
    current_kv_list->shape[0] += current_info_kv_list->GetElementNumber();
    MemcpyAsync(current_info_kv_list->GetPtr<void>(), kv_list_host.data(),
                kv_list_host.size() * sizeof(decltype(kv_list_host)::value_type), MEMCPY_HOST_TO_DEVICE,
                config.context->device->GetD2HStream());

#ifdef ENABLE_CUDA
    if (!host_fill_all_layers && total_block_num > 0 && config.layer_num_on_node > 1) {
      InvokeProcessKvList(current_info_kv_list->GetPtr<void*>(), config.layer_num_on_node, total_block_num * 2,
                          config.block_size, config.context->device->GetD2HStream().Get());
    }
#endif
  }

#ifdef ENABLE_ACL
  StreamSynchronize(config.context->device->GetD2HStream());
#endif
  EventRecord(config.kvcache_offset_event, config.context->device->GetD2HStream());
}

void PrepareKVCacheBlockTable(StandardAttentionMeta& meta, StandardSharedAttentionMeta& shared_meta,
                              const AttentionBuilderConfig& config, int layer_kv_cache_ptr_layer_factor,
                              int layer_kv_cache_ptr_kv_factor) {
  PROFILE_EVENT_SCOPE(PrepareKVCacheBlockTable, "PrepareKVCacheBlockTable", config.rank);
  if (!config.enable_blocked_multi_token_forwarding_kv && !config.use_flashinfer_for_decode) {
    return;
  }

  const auto& reqs = meta.dp_reqs;

  if (!shared_meta.layer_kv_cache_ptr_initialized) {
    shared_meta.layer_kv_cache_ptr_initialized = true;

    const auto block_allocator_manager = reqs.front()->block_allocator_manager;
    const auto block_allocator_group = block_allocator_manager->GetBlockAllocatorGroup(config.attn_dp_group_id);
    const auto device_allocator = block_allocator_group->GetDeviceBlockAllocator(config.attn_dp_rank_id);

    for (const auto& kv_cache_type : config.kv_cache_types) {
      void** current_layer_kv_cache_ptr;
      if (kv_cache_type == "attention") {
        current_layer_kv_cache_ptr = shared_meta.layer_kv_cache_ptr.GetPtr<void*>();
      } else {  // kv_cache_type == "indexer"
        current_layer_kv_cache_ptr = shared_meta.GetDSA()->layer_indexer_kv_cache_ptr.GetPtr<void*>();
      }
      *reinterpret_cast<int64_t*>(current_layer_kv_cache_ptr) =
          block_allocator_group->GetBlockAllocatorGroupConfig().device_block_num * layer_kv_cache_ptr_layer_factor *
          layer_kv_cache_ptr_kv_factor;
      ++current_layer_kv_cache_ptr;

      void* const blocks_base_ptr = device_allocator->GetBlocksBasePtr();
      const auto& cached_layer_offsets = config.kv_cache_layer_offsets.at(kv_cache_type);
      for (int layer_idx = 0; layer_idx < config.layer_num_on_node; ++layer_idx) {
        const auto& layer_offsets = cached_layer_offsets[layer_idx];
        *current_layer_kv_cache_ptr++ = blocks_base_ptr + layer_offsets.k_offset;
        *current_layer_kv_cache_ptr++ = blocks_base_ptr + layer_offsets.v_offset;
      }
    }
  }

  size_t max_num_blocks_per_query = 0;
  for (const auto& req : reqs) {
    max_num_blocks_per_query =
        std::max(max_num_blocks_per_query, req->atb_kv_cache_base_blk_ids[config.attn_dp_rank_id].size());
  }

  std::vector<int32_t> block_table_host(reqs.size() * max_num_blocks_per_query, -1);
  for (size_t i = 0; i < reqs.size(); ++i) {
    const auto& block_ids = reqs[i]->atb_kv_cache_base_blk_ids[config.attn_dp_rank_id];
    for (size_t base_block_idx = 0; base_block_idx < block_ids.size(); ++base_block_idx) {
      block_table_host[i * max_num_blocks_per_query + base_block_idx] = block_ids[base_block_idx];
    }
  }
  KLLM_LOG_DEBUG << "block_table_host " << block_table_host;
  meta.block_table =
      shared_meta.block_table.GetView({reqs.size(), max_num_blocks_per_query}, shared_meta.block_table.shape[0]);
  shared_meta.block_table.shape[0] += meta.block_table.GetElementNumber();
  MemcpyAsync(meta.block_table.GetPtr<void>(), block_table_host.data(),
              block_table_host.size() * sizeof(decltype(block_table_host)::value_type), MEMCPY_HOST_TO_DEVICE,
              config.context->device->GetH2DStream());

#ifdef ENABLE_ACL
  StreamSynchronize(config.context->device->GetD2HStream());
#endif
  EventRecord(config.kvcache_offset_event, config.context->device->GetD2HStream());
}

}  // namespace attention_meta_builder_detail
}  // namespace ksana_llm
