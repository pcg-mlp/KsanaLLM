/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v4/attention_meta/deepseek_v4_mqa_builder.h"

#include <fmt/core.h>

#include "configure/environment.h"
#include "device/device_utils.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

namespace {

// ── Helper: physical-position calculators ────────────────────────────────────

// Convert a logical token position to its physical cache position in the main KV pool.
inline int GetPhysicalTokenPos(const std::vector<int32_t>& block_ids, int token_pos, int compress_ratio,
                               int block_token_num) {
  return token_pos < 0 ? -1
                       : block_ids[token_pos / block_token_num] * (block_token_num / compress_ratio) +
                             (token_pos / compress_ratio) % (block_token_num / compress_ratio);
}

// Map a logical token position to its physical position in the SWA pool.
// Returns -1 when the position falls outside the allocated SWA window (kernel skips -1).
inline int GetSwaPhysicalTokenPos(const std::vector<int>& swa_block_ids,
                                  const std::vector<size_t>& swa_block_idx_offsets, int token_pos,
                                  int block_token_num) {
  if (token_pos < 0) return -1;
  if (swa_block_idx_offsets.empty()) return -1;
  const int global_block_idx = token_pos / block_token_num;
  const int local_swa_idx = global_block_idx - static_cast<int>(swa_block_idx_offsets[0]);
  if (local_swa_idx < 0 || local_swa_idx >= static_cast<int>(swa_block_ids.size())) return -1;
  return swa_block_ids[local_swa_idx] * block_token_num + (token_pos % block_token_num);
}

// ── Helper: one-time SWA / tail base-pointer init ────────────────────────────

// Resolve the per-layer SWA / tail base pointers from the first request that has a valid
// swa_block_base_addr.  Guarded by layer_kv_cache_ptr_initialized; safe to call from both
// PrepareMqaFlash and PrepareMqaPaged so that pure-decode steps are also covered.
template <typename ReqPtrContainer>
void InitSwaBasePointersIfNeeded(MqaSharedAttentionMeta& mqa_shared, const ReqPtrContainer& dp_reqs) {
  if (mqa_shared.layer_kv_cache_ptr_initialized) return;

  void* swa_pool_base = nullptr;
  int64_t swa_pool_block_num = 0;
  for (const auto* req : dp_reqs) {
    if (req->swa_block_base_addr != nullptr) {
      swa_pool_base = req->swa_block_base_addr;
      swa_pool_block_num = req->swa_pool_block_num;
      break;
    }
  }
  KLLM_CHECK_WITH_INFO(swa_pool_base != nullptr,
                       "PrepareMqa: forward_req.swa_block_base_addr is nullptr "
                       "(SWA pool not enabled or model_builder did not populate it).");
  KLLM_CHECK_WITH_INFO(swa_pool_block_num > 0, "PrepareMqa: swa_pool_block_num must be > 0.");

  mqa_shared.layer_kv_cache_ptr_initialized = true;

  int64_t* swa_meta = mqa_shared.layer_swa_kv_cache_ptr_tensor.GetPtr<int64_t>();
  int64_t* tail_attn_meta = mqa_shared.layer_tail_attn_cache_ptr_tensor.GetPtr<int64_t>();
  int64_t* tail_indexer_meta = mqa_shared.layer_tail_indexer_cache_ptr_tensor.GetPtr<int64_t>();
  swa_meta[0] = swa_pool_block_num;
  // swa_meta[1] (swa_block_size) was written during Init(); leave it unchanged.
  tail_attn_meta[0] = swa_pool_block_num;
  tail_indexer_meta[0] = swa_pool_block_num;

  void** layer_swa_ptr = mqa_shared.layer_swa_kv_cache_ptr_tensor.GetPtr<void*>() + 2;
  void** layer_tail_attn_ptr = mqa_shared.layer_tail_attn_cache_ptr_tensor.GetPtr<void*>() + 1;
  void** layer_tail_indexer_ptr = mqa_shared.layer_tail_indexer_cache_ptr_tensor.GetPtr<void*>() + 1;
  uint8_t* const cache_base_ptr = static_cast<uint8_t*>(swa_pool_base);

  KLLM_LOG_INFO << fmt::format(
      "init MQA SWA pool base_ptr={} swa_pool_block_num={} swa_block_size={} block_size(main)={}",
      static_cast<const void*>(cache_base_ptr), swa_pool_block_num, mqa_shared.swa_block_size, mqa_shared.block_size);

  for (int layer_idx = 0; layer_idx < mqa_shared.layer_num_on_node; ++layer_idx) {
    *layer_swa_ptr = cache_base_ptr + reinterpret_cast<size_t>(*layer_swa_ptr);
    *layer_tail_attn_ptr = cache_base_ptr + reinterpret_cast<size_t>(*layer_tail_attn_ptr);
    *layer_tail_indexer_ptr = cache_base_ptr + reinterpret_cast<size_t>(*layer_tail_indexer_ptr);
    KLLM_LOG_DEBUG << fmt::format("layer {} swa_base={} tail_attn_base={} tail_indexer_base={}", layer_idx,
                                  *layer_swa_ptr, *layer_tail_attn_ptr, *layer_tail_indexer_ptr);
    ++layer_swa_ptr;
    ++layer_tail_attn_ptr;
    ++layer_tail_indexer_ptr;
  }
}

// ── Helper: per-batch prepare subfunctions ───────────────────────────────────

template <typename ReqPtrContainer>
void PrepareOutLoc(MqaAttentionMeta& mqa_meta, MqaSharedAttentionMeta& mqa_shared, const ReqPtrContainer& dp_reqs,
                   const AttentionBuilderConfig& config) {
  const int block_token_num = mqa_shared.block_token_num;
  for (const auto* req : dp_reqs) {
    const int extend_len = req->GetInputIdsLength();
    const int seq_len = static_cast<int>(req->forwarding_tokens->size());
    const int prefix_len = seq_len - extend_len;
    const auto& swa_block_ids = req->swa_kv_cache_blocks;
    const auto& swa_block_idx_offsets = req->swa_kv_cache_block_idx_offsets;
    for (int i = prefix_len; i < seq_len; ++i) {
      mqa_meta.raw_out_loc_host.push_back(
          GetSwaPhysicalTokenPos(swa_block_ids, swa_block_idx_offsets, i, block_token_num));
    }
  }

  mqa_meta.raw_out_loc_tensor =
      mqa_shared.raw_out_loc_tensor.GetView({mqa_meta.raw_out_loc_host.size()}, mqa_shared.raw_out_loc_tensor.shape[0]);
  mqa_shared.raw_out_loc_tensor.shape[0] += mqa_meta.raw_out_loc_tensor.GetElementNumber();
  MemcpyAsync(mqa_meta.raw_out_loc_tensor.GetPtr<void>(), mqa_meta.raw_out_loc_host.data(),
              mqa_meta.raw_out_loc_host.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE,
              config.context->device->GetH2DStream());
}

template <typename ReqPtrContainer>
void PrepareBlockIndices(MqaAttentionMeta& mqa_meta, MqaSharedAttentionMeta& mqa_shared, const ReqPtrContainer& dp_reqs,
                         size_t total_dp_input_ids_len, const AttentionBuilderConfig& config) {
  const int block_token_num = mqa_shared.block_token_num;
  const int attn_dp_rank_id = static_cast<int>(mqa_shared.attn_dp_rank_id);
  constexpr int kSwaSize = static_cast<int>(MqaSharedAttentionMeta::kSwaSize);
  constexpr int kC128Ratio = static_cast<int>(MqaSharedAttentionMeta::kC128Ratio);
  constexpr int kTopkBlockSize = static_cast<int>(MqaSharedAttentionMeta::kTopkBlockSize);

  int c128_max_seq_len_clamp1 = 1;
  for (const auto* req : dp_reqs) {
    const int c128_seq_len = static_cast<int>(req->forwarding_tokens->size()) / kC128Ratio;
    if (c128_seq_len > c128_max_seq_len_clamp1) c128_max_seq_len_clamp1 = c128_seq_len;
  }
  const int aligned_c128_max = RoundUp(c128_max_seq_len_clamp1, kTopkBlockSize);

  for (const auto* req : dp_reqs) {
    const int extend_len = req->GetInputIdsLength();
    const int seq_len = static_cast<int>(req->forwarding_tokens->size());
    const int prefix_len = seq_len - extend_len;
    const auto& block_ids = req->atb_kv_cache_base_blk_ids[attn_dp_rank_id];
    const auto& swa_block_ids = req->swa_kv_cache_blocks;
    const auto& swa_block_idx_offsets = req->swa_kv_cache_block_idx_offsets;

    for (int i = prefix_len; i < seq_len; ++i) {
      for (int j = i - kSwaSize + 1; j <= i; ++j) {
        mqa_meta.swa_block_indices_host.push_back(
            GetSwaPhysicalTokenPos(swa_block_ids, swa_block_idx_offsets, j, block_token_num));
      }
      int c128_len = 0;
      for (int j = kC128Ratio - 1; j <= i; j += kC128Ratio) {
        mqa_meta.c128_block_indices_host.push_back(GetPhysicalTokenPos(block_ids, j, kC128Ratio, block_token_num));
        ++c128_len;
      }
      while (c128_len < aligned_c128_max) {
        mqa_meta.c128_block_indices_host.push_back(-1);
        ++c128_len;
      }
    }
  }

  mqa_meta.swa_block_indices_tensor = mqa_shared.swa_block_indices_tensor.GetView(
      {mqa_meta.swa_block_indices_host.size()}, mqa_shared.swa_block_indices_tensor.shape[0]);
  mqa_shared.swa_block_indices_tensor.shape[0] += mqa_meta.swa_block_indices_tensor.GetElementNumber();
  MemcpyAsync(mqa_meta.swa_block_indices_tensor.GetPtr<void>(), mqa_meta.swa_block_indices_host.data(),
              mqa_meta.swa_block_indices_host.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE,
              config.context->device->GetH2DStream());

  mqa_meta.c128_block_indices_tensor = mqa_shared.c128_block_indices_tensor.GetView(
      {mqa_meta.c128_block_indices_host.size()}, mqa_shared.c128_block_indices_tensor.shape[0]);
  mqa_shared.c128_block_indices_tensor.shape[0] += mqa_meta.c128_block_indices_tensor.GetElementNumber();
  MemcpyAsync(mqa_meta.c128_block_indices_tensor.GetPtr<void>(), mqa_meta.c128_block_indices_host.data(),
              mqa_meta.c128_block_indices_host.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE,
              config.context->device->GetH2DStream());

  mqa_meta.swa_block_indices_tensor.shape = {total_dp_input_ids_len, MqaSharedAttentionMeta::kSwaSize};
  mqa_meta.c128_block_indices_tensor.shape = {total_dp_input_ids_len, static_cast<size_t>(aligned_c128_max)};
}

template <typename ReqPtrContainer>
void PrepareTopkLengths(MqaAttentionMeta& mqa_meta, MqaSharedAttentionMeta& mqa_shared, const ReqPtrContainer& dp_reqs,
                        const AttentionBuilderConfig& config) {
  constexpr int kC128Ratio = static_cast<int>(MqaSharedAttentionMeta::kC128Ratio);
  for (const auto* req : dp_reqs) {
    const int extend_len = req->GetInputIdsLength();
    const int seq_len = static_cast<int>(req->forwarding_tokens->size());
    const int c128_topk_len = seq_len / kC128Ratio;
    for (int i = 0; i < extend_len; ++i) {
      mqa_meta.c128_topk_lengths_host.push_back(c128_topk_len);
    }
  }

  mqa_meta.c128_topk_lengths_tensor = mqa_shared.c128_topk_lengths_tensor.GetView(
      {mqa_meta.c128_topk_lengths_host.size()}, mqa_shared.c128_topk_lengths_tensor.shape[0]);
  mqa_shared.c128_topk_lengths_tensor.shape[0] += mqa_meta.c128_topk_lengths_tensor.GetElementNumber();
  MemcpyAsync(mqa_meta.c128_topk_lengths_tensor.GetPtr<void>(), mqa_meta.c128_topk_lengths_host.data(),
              mqa_meta.c128_topk_lengths_host.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE,
              config.context->device->GetH2DStream());
}

void AssignTileSchedulerAndNumSplits(MqaAttentionMeta& mqa_meta, MqaSharedAttentionMeta& mqa_shared, size_t q_seq_len,
                                     size_t batch_size, bool is_flash) {
#ifdef ENABLE_CUDA
  const size_t meta_idx = is_flash ? mqa_shared.c4_tile_scheduler_metadatas.size() - 1 : q_seq_len - 1;
  mqa_meta.c4_tile_scheduler_metadata = mqa_shared.c4_tile_scheduler_metadatas[meta_idx];
  mqa_meta.c4_tile_scheduler_metadata.shape[2] = 0;  // mark as uninitialized; kernel fills it
  mqa_meta.c128_tile_scheduler_metadata = mqa_shared.c128_tile_scheduler_metadatas[meta_idx];
  mqa_meta.c128_tile_scheduler_metadata.shape[2] = 0;

  mqa_meta.c4_num_splits =
      mqa_shared.c4_num_splits_tensor.GetView({batch_size + 1}, mqa_shared.c4_num_splits_tensor.shape[0]);
  mqa_shared.c4_num_splits_tensor.shape[0] += mqa_meta.c4_num_splits.GetElementNumber();
  mqa_meta.c128_num_splits =
      mqa_shared.c128_num_splits_tensor.GetView({batch_size + 1}, mqa_shared.c128_num_splits_tensor.shape[0]);
  mqa_shared.c128_num_splits_tensor.shape[0] += mqa_meta.c128_num_splits.GetElementNumber();
#endif
}

}  // namespace

// ── Public API ───────────────────────────────────────────────────────────────

void PrepareMqaFlash(MqaAttentionMeta& mqa_meta, MqaSharedAttentionMeta& mqa_shared,
                     const FlashAttentionMeta& flash_meta, const AttentionBuilderConfig& config) {
  if (flash_meta.dp_reqs.empty()) return;

  mqa_meta.Reset();

  InitSwaBasePointersIfNeeded(mqa_shared, flash_meta.dp_reqs);

  PrepareOutLoc(mqa_meta, mqa_shared, flash_meta.dp_reqs, config);
  PrepareBlockIndices(mqa_meta, mqa_shared, flash_meta.dp_reqs, flash_meta.total_dp_input_ids_len, config);
  PrepareTopkLengths(mqa_meta, mqa_shared, flash_meta.dp_reqs, config);

  // Flash uses the extra entry at the back (index == decode_token_num_threshold)
  const size_t flash_batch = flash_meta.dp_reqs.size();
  AssignTileSchedulerAndNumSplits(mqa_meta, mqa_shared, /*q_seq_len=*/0, flash_batch, /*is_flash=*/true);
}

void PrepareMqaPaged(MqaAttentionMeta& mqa_meta, MqaSharedAttentionMeta& mqa_shared,
                     const PagedAttentionMeta& paged_meta, const AttentionBuilderConfig& config) {
  if (paged_meta.dp_reqs.empty()) return;

  mqa_meta.Reset();

  InitSwaBasePointersIfNeeded(mqa_shared, paged_meta.dp_reqs);

  PrepareOutLoc(mqa_meta, mqa_shared, paged_meta.dp_reqs, config);
  PrepareBlockIndices(mqa_meta, mqa_shared, paged_meta.dp_reqs, paged_meta.total_dp_input_ids_len, config);
  PrepareTopkLengths(mqa_meta, mqa_shared, paged_meta.dp_reqs, config);

  const size_t paged_batch = paged_meta.dp_reqs.size();
  AssignTileSchedulerAndNumSplits(mqa_meta, mqa_shared, paged_meta.q_seq_len, paged_batch, /*is_flash=*/false);
}

}  // namespace ksana_llm
