/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/attention_meta/flash_attention_meta_builder.h"

#include <algorithm>
#include <numeric>

namespace ksana_llm {

using namespace attention_meta_builder_detail;

StandardFlashAttentionMetaBuilder::StandardFlashAttentionMetaBuilder(const AttentionBuilderConfig& config)
    : config_(config) {}

void StandardFlashAttentionMetaBuilder::Build(FlashAttentionMeta& meta, SharedAttentionMeta& shared_meta,
                                              size_t dp_total_prefix_len) {
  if (meta.use_cache) {
    auto& standard_shared = *shared_meta.GetStandard();
    auto& standard_meta = static_cast<StandardAttentionMeta&>(meta);
    PrepareFlexibleCache(meta, shared_meta);
    PrepareKVCacheBlocks(standard_meta, standard_shared, config_);
    OnPrepareKVCacheBlockTable(standard_meta, standard_shared);
  }

  PrepareFlashRotary(meta, shared_meta, dp_total_prefix_len);
}

void StandardFlashAttentionMetaBuilder::OnPrepareKVCacheBlockTable(StandardAttentionMeta& meta,
                                                                   StandardSharedAttentionMeta& shared_meta) {
  PrepareKVCacheBlockTable(meta, shared_meta, config_, config_.layer_num_on_node, 2);
}

void StandardFlashAttentionMetaBuilder::PrepareFlexibleCache(FlashAttentionMeta& meta,
                                                             SharedAttentionMeta& shared_meta) {
  auto& standard_shared = *shared_meta.GetStandard();
  // Flexible cache supported scope: dense model with paged attn layout, deepseek model
  // Flexible cache not supported scope: dense model with flash attn layout
  if (!config_.runtime_config.enable_flexible_caching && config_.enable_blocked_multi_token_forwarding_kv) {
    return;
  }

  std::vector<int> dst_flexible_kv_cache_id_cpu;
  std::vector<int> src_flexible_kv_cache_id_cpu;
  std::vector<int> dst_flexible_token_idx_cpu;
  std::vector<int> src_flexible_token_idx_cpu;
  std::vector<uint64_t> flexible_offset_uint64_cpu = {0};

  for (const auto& req : meta.dp_reqs) {
    if (req->attn_dp_group_id != config_.attn_dp_group_id) {
      continue;
    }

    flexible_offset_uint64_cpu.emplace_back(flexible_offset_uint64_cpu.back() + req->prefix_cache_len -
                                            req->flexible_cache_len);
    for (const auto& task : *req->flexible_cached_copy_tasks) {
      dst_flexible_kv_cache_id_cpu.emplace_back(task.dst_block_id_);
      src_flexible_kv_cache_id_cpu.emplace_back(task.src_block_id_);
      dst_flexible_token_idx_cpu.emplace_back(task.dst_token_idx_);
      src_flexible_token_idx_cpu.emplace_back(task.src_token_idx_);
    }
  }

  standard_shared.dp_flexible_offset_uint64_tensor.shape = {flexible_offset_uint64_cpu.size()};
  MemcpyAsync(standard_shared.dp_flexible_offset_uint64_tensor.GetPtr<void>(), flexible_offset_uint64_cpu.data(),
              flexible_offset_uint64_cpu.size() * sizeof(decltype(flexible_offset_uint64_cpu)::value_type),
              MEMCPY_HOST_TO_DEVICE, config_.context->device->GetH2DStream());

  if (dst_flexible_kv_cache_id_cpu.empty() || !config_.runtime_config.enable_flexible_caching) {
    standard_shared.dp_dst_flexible_kv_cache_tensor.shape = {0};
    return;
  }

  std::vector<void*> dst_flexible_kv_cache_cpu(dst_flexible_kv_cache_id_cpu.size());
  std::vector<void*> src_flexible_kv_cache_cpu(src_flexible_kv_cache_id_cpu.size());
  const auto block_allocator_manager = meta.dp_reqs.front()->block_allocator_manager;
  auto device_allocator = block_allocator_manager->GetBlockAllocatorGroup(config_.attn_dp_group_id)
                              ->GetDeviceBlockAllocator(config_.attn_dp_rank_id);
  device_allocator->GetBlockPtrs(dst_flexible_kv_cache_id_cpu, dst_flexible_kv_cache_cpu);
  device_allocator->GetBlockPtrs(src_flexible_kv_cache_id_cpu, src_flexible_kv_cache_cpu);

  standard_shared.dp_dst_flexible_kv_cache_tensor.shape = {dst_flexible_kv_cache_cpu.size()};
  MemcpyAsync(standard_shared.dp_dst_flexible_kv_cache_tensor.GetPtr<void>(), dst_flexible_kv_cache_cpu.data(),
              dst_flexible_kv_cache_cpu.size() * sizeof(decltype(dst_flexible_kv_cache_cpu)::value_type),
              MEMCPY_HOST_TO_DEVICE, config_.context->device->GetH2DStream());
  MemcpyAsync(standard_shared.dp_src_flexible_kv_cache_tensor.GetPtr<void>(), src_flexible_kv_cache_cpu.data(),
              src_flexible_kv_cache_cpu.size() * sizeof(decltype(src_flexible_kv_cache_cpu)::value_type),
              MEMCPY_HOST_TO_DEVICE, config_.context->device->GetH2DStream());
  MemcpyAsync(standard_shared.dp_dst_flexible_token_idx_tensor.GetPtr<void>(), dst_flexible_token_idx_cpu.data(),
              dst_flexible_token_idx_cpu.size() * sizeof(decltype(dst_flexible_token_idx_cpu)::value_type),
              MEMCPY_HOST_TO_DEVICE, config_.context->device->GetH2DStream());
  MemcpyAsync(standard_shared.dp_src_flexible_token_idx_tensor.GetPtr<void>(), src_flexible_token_idx_cpu.data(),
              src_flexible_token_idx_cpu.size() * sizeof(decltype(src_flexible_token_idx_cpu)::value_type),
              MEMCPY_HOST_TO_DEVICE, config_.context->device->GetH2DStream());
}

void StandardFlashAttentionMetaBuilder::PrepareFlashRotary(FlashAttentionMeta& meta, SharedAttentionMeta& shared_meta,
                                                           size_t dp_total_prefix_len) {
  auto& standard_shared = *shared_meta.GetStandard();
  size_t total_input_len = 0;
  for (const auto& req : meta.dp_reqs) {
    total_input_len += req->forwarding_tokens->size();
  }

  const size_t rotary_host_size =
      config_.enable_blocked_multi_token_forwarding_kv ? meta.total_dp_input_ids_len : total_input_len;
  std::vector<int64_t> rotary_mask_host(rotary_host_size, 1);
  std::vector<int64_t> rotary_pos_host(rotary_host_size);

  size_t rotary_host_idx = 0;
  for (const auto& req : meta.dp_reqs) {
    if (config_.enable_blocked_multi_token_forwarding_kv) {
      std::iota(rotary_pos_host.begin() + rotary_host_idx,
                rotary_pos_host.begin() + rotary_host_idx + req->forwarding_tokens->size() - req->prefix_cache_len,
                req->prefix_cache_len);
      rotary_host_idx += req->forwarding_tokens->size() - req->prefix_cache_len;
    } else {
      // mask real prefix(exclude flexible cache), now equals kv_cached_token_num
      std::fill_n(rotary_mask_host.begin() + rotary_host_idx, req->kv_cached_token_num, 0);
      // Assign rotary positional values
      std::iota(rotary_pos_host.begin() + rotary_host_idx,
                rotary_pos_host.begin() + rotary_host_idx + req->forwarding_tokens->size(), 0);
      rotary_host_idx += req->forwarding_tokens->size();
    }
  }

  meta.rotary_embedding_pos = standard_shared.rotary_embedding_pos.GetView(
      {rotary_pos_host.size()}, standard_shared.rotary_embedding_pos.shape[0]);
  standard_shared.rotary_embedding_pos.shape[0] += meta.rotary_embedding_pos.GetElementNumber();
  MemcpyAsync(meta.rotary_embedding_pos.GetPtr<void>(), rotary_pos_host.data(),
              sizeof(decltype(rotary_pos_host)::value_type) * rotary_host_idx, MEMCPY_HOST_TO_DEVICE,
              config_.context->device->GetH2DStream());
  meta.rotary_embedding_mask = standard_shared.rotary_embedding_mask.GetView(
      {rotary_mask_host.size()}, standard_shared.rotary_embedding_mask.shape[0]);
  standard_shared.rotary_embedding_mask.shape[0] += meta.rotary_embedding_mask.GetElementNumber();
  MemcpyAsync(meta.rotary_embedding_mask.GetPtr<void>(), rotary_mask_host.data(),
              sizeof(decltype(rotary_mask_host)::value_type) * rotary_host_idx, MEMCPY_HOST_TO_DEVICE,
              config_.context->device->GetH2DStream());

  // For DeepSeek model, rope of flexible cached tokens and forwarding new tokens cannot be fused
  // because q&k buffers differ in total_tokens when there is prefix
  // If q&k forwarding parts is processed together, k flexible part must be handled separately
  if (standard_shared.dp_dst_flexible_kv_cache_tensor.shape[0] != 0) {
    // rotary_mask_host can be shared for rotary_pos_host and mla_dst_flexible_rotary_pos_host
    rotary_mask_host.assign(total_input_len, 0);
    // rotary_pos_host stores reverse rope information of flexible cached tokens
    rotary_pos_host.assign(total_input_len, 0);
    // mla_dst_flexible_rotary_pos_host stores correct rope information of flexible cached tokens
    std::vector<int64_t> mla_dst_flexible_rotary_pos_host(total_input_len, 0);

    size_t flexible_rotary_idx = 0;
    for (const auto& req : meta.dp_reqs) {
      if (req->flexible_cache_len > 0) {
        std::fill(rotary_mask_host.begin() + flexible_rotary_idx + req->prefix_cache_len - req->flexible_cache_len,
                  rotary_mask_host.begin() + flexible_rotary_idx + req->prefix_cache_len, 1);
        for (auto& task : *req->flexible_cached_copy_tasks) {
          // Reverse rope information stores src_token_idx_ to clear existing rope information
          rotary_pos_host[flexible_rotary_idx + task.dst_token_idx_] = task.src_token_idx_;
          // For DeepSeek model, prepare correct rope information of flexible part
          if (config_.model_config.use_mla) {
            // Correct rope information stores dst_token_idx_ to embed correct rope information
            mla_dst_flexible_rotary_pos_host[flexible_rotary_idx + task.dst_token_idx_] = task.dst_token_idx_;
          }
        }
      }
      flexible_rotary_idx += req->forwarding_tokens->size();
    }
    MemcpyAsync(standard_shared.dp_flexible_rotary_embedding_mask.GetPtr<void>(), rotary_mask_host.data(),
                sizeof(decltype(rotary_mask_host)::value_type) * flexible_rotary_idx, MEMCPY_HOST_TO_DEVICE,
                config_.context->device->GetH2DStream());
    MemcpyAsync(standard_shared.dp_src_flexible_rotary_embedding_pos.GetPtr<void>(), rotary_pos_host.data(),
                sizeof(decltype(rotary_pos_host)::value_type) * flexible_rotary_idx, MEMCPY_HOST_TO_DEVICE,
                config_.context->device->GetH2DStream());
    if (config_.model_config.use_mla) {
      MemcpyAsync(standard_shared.dp_dst_flexible_rotary_embedding_pos.GetPtr<void>(),
                  mla_dst_flexible_rotary_pos_host.data(),
                  sizeof(decltype(mla_dst_flexible_rotary_pos_host)::value_type) * flexible_rotary_idx,
                  MEMCPY_HOST_TO_DEVICE, config_.context->device->GetH2DStream());
    }
  }
  EventRecord(config_.rotary_embedding_event, config_.context->device->GetH2DStream());
#if defined(ENABLE_ACL) || defined(ENABLE_KUNLUN)
  // KUNLUN deliberately keeps the host sync here even though PrepareDecodeRotary dropped it. The
  // asymmetry is intentional, not an oversight:
  //   - every H2D above sources from a function local vector (rotary_pos_host, rotary_mask_host,
  //     mla_dst_flexible_rotary_pos_host). Dropping the sync would let those buffers be freed while
  //     their DMA is still in flight, since xpu_memcpy_async gives no lifetime guarantee for pageable
  //     host memory. The decode path only got to drop its sync after its staging buffers were made
  //     persistent members (decode_rotary_mask_ones / decode_rotary_pos_host).
  //   - prefill runs once per request batch and is not on the XPU-graph replay path, so the sync costs
  //     far less here than the per-step one that was removed from decode.
  // If this ever needs to go away, migrate the three buffers above to persistent members first.
  StreamSynchronize(config_.context->device->GetH2DStream());
#endif
}

}  // namespace ksana_llm
