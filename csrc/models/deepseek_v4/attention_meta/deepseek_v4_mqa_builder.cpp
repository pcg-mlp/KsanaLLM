/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v4/attention_meta/deepseek_v4_mqa_builder.h"

#include <fmt/core.h>

#include <algorithm>

#include "configure/environment.h"
#include "device/device_utils.h"
#include "models/deepseek_v4/attention_meta/prepare_dsv4_mqa_meta_dispatch.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

namespace {

// ── Helper: physical-position calculators ────────────────────────────────────

// Map a logical token position to its physical slot in the SWA pool.
// Assumes swa_kv_cache_block_idx_offsets are contiguous (offsets[i] == offsets[0] + i).
// Returns -1 when token_pos falls outside the allocated SWA window (kernel skips -1).
inline int GetSwaPoolTokenPos(const std::vector<int>& swa_block_ids, const std::vector<size_t>& swa_block_idx_offsets,
                              int token_pos, int block_token_num) {
  if (token_pos < 0) return -1;
  if (swa_block_idx_offsets.empty()) return -1;
  const int global_block_idx = token_pos / block_token_num;
  const int local_swa_idx = global_block_idx - static_cast<int>(swa_block_idx_offsets[0]);
  if (local_swa_idx < 0 || local_swa_idx >= static_cast<int>(swa_block_ids.size())) return -1;
  return swa_block_ids[local_swa_idx] * block_token_num + (token_pos % block_token_num);
}

// Map a logical token position to its slot in the SWA cache buffer.
// The buffer covers FA block indices [swa_start_block_idx, end_block_idx] contiguously.
// Buffer slot for FA block g = req_buffer_start + (g - swa_start_block_idx).
// Returns -1 when token_pos maps to a FA block index below swa_start_block_idx.
inline int GetSwaBufferTokenPos(int token_pos, int block_token_num, int req_buffer_start, int swa_start_block_idx) {
  if (token_pos < 0) return -1;
  const int global_block_idx = token_pos / block_token_num;
  const int buffer_local_block = global_block_idx - swa_start_block_idx;
  if (buffer_local_block < 0) return -1;
  return (req_buffer_start + buffer_local_block) * block_token_num + (token_pos % block_token_num);
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

// ── Helper: SWA cache buffer range ───────────────────────────────────────────

// Describes the contiguous FA-block range that the SWA cache buffer must cover for one request.
struct SwaCacheBufferRange {
  int swa_start_blk;  // first FA block in the SWA window: max(kv_cached - kSwaSize, 0) / block_token_num
  int kv_cached_blk;  // first FA block containing new (extend) tokens: kv_cached / block_token_num
  int end_blk;        // last FA block in forwarding_tokens: (seq_len - 1) / block_token_num
  int slots;          // total buffer slots for this request: end_blk - swa_start_blk + 1
};

inline SwaCacheBufferRange ComputeSwaCacheBufferRange(int kv_cached, int seq_len, int block_token_num) {
  constexpr int kSwaSize = static_cast<int>(MqaSharedAttentionMeta::kSwaSize);
  SwaCacheBufferRange r;
  r.swa_start_blk = std::max(kv_cached - kSwaSize, 0) / block_token_num;
  r.kv_cached_blk = kv_cached / block_token_num;
  r.end_blk = (seq_len - 1) / block_token_num;
  r.slots = r.end_blk - r.swa_start_blk + 1;
  return r;
}

// ── Helper: per-batch prepare subfunctions ───────────────────────────────────

template <typename ReqPtrContainer>
void PrepareOutLoc(MqaAttentionMeta& mqa_meta, MqaSharedAttentionMeta& mqa_shared, const ReqPtrContainer& dp_reqs,
                   const AttentionBuilderConfig& config) {
  const int block_token_num = mqa_shared.block_token_num;
  int req_buffer_start = mqa_meta.has_non_contiguous_swa ? 0 : -1;
  for (const auto* req : dp_reqs) {
    const int extend_len = req->GetInputIdsLength();
    const int seq_len = static_cast<int>(req->forwarding_tokens->size());
    const int prefix_len = seq_len - extend_len;
    const auto& swa_block_ids = req->swa_kv_cache_blocks;
    const auto& swa_block_idx_offsets = req->swa_kv_cache_block_idx_offsets;
    if (req_buffer_start >= 0) {
      const auto range = ComputeSwaCacheBufferRange(req->kv_cached_token_num, seq_len, block_token_num);
      for (int i = prefix_len; i < seq_len; ++i) {
        mqa_meta.raw_out_loc_host.push_back(
            GetSwaBufferTokenPos(i, block_token_num, req_buffer_start, range.swa_start_blk));
      }
      req_buffer_start += range.slots;
    } else {
      for (int i = prefix_len; i < seq_len; ++i) {
        mqa_meta.raw_out_loc_host.push_back(
            GetSwaPoolTokenPos(swa_block_ids, swa_block_idx_offsets, i, block_token_num));
      }
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
void PrepareBlockIndicesAndTopkLengths(MqaAttentionMeta& mqa_meta, MqaSharedAttentionMeta& mqa_shared,
                                       const ReqPtrContainer& dp_reqs, size_t total_dp_input_ids_len,
                                       const StandardAttentionMeta& standard_meta, const AttentionBuilderConfig& config,
                                       bool is_flash) {
  const size_t batch_size = dp_reqs.size();
  constexpr int kTailStride = MqaSharedAttentionMeta::kSwaSize - 1;

  mqa_shared.cu_extend_lens_host.assign(1, 0);
  mqa_shared.cu_prefix_lens_host.assign(1, 0);
  mqa_shared.swa_prefix_tail_block_indices_host.assign(batch_size * MqaSharedAttentionMeta::kSwaSize, -1);

  int max_extend_len = 0;
  int c128_max_seq_len_clamp1 = 1;
  int req_buffer_start = mqa_meta.has_non_contiguous_swa ? 0 : -1;
  size_t r = 0;
  for (const auto* req : dp_reqs) {
    const int extend_len = req->GetInputIdsLength();
    const int seq_len = static_cast<int>(req->forwarding_tokens->size());
    const int c128_seq_len = seq_len / static_cast<int>(MqaSharedAttentionMeta::kC128Ratio);
    const int prefix_len = seq_len - extend_len;

    if (extend_len > max_extend_len) {
      max_extend_len = extend_len;
    }
    if (c128_seq_len > c128_max_seq_len_clamp1) {
      c128_max_seq_len_clamp1 = c128_seq_len;
    }

    mqa_shared.cu_extend_lens_host.push_back(mqa_shared.cu_extend_lens_host.back() + extend_len);
    mqa_shared.cu_prefix_lens_host.push_back(mqa_shared.cu_prefix_lens_host.back() + prefix_len);

    // Per-req SWA prefix-tail row: kTailStride entries covering [prefix_len - kTailStride, prefix_len).
    const int tail_first_j = prefix_len - kTailStride;
    int* row = mqa_shared.swa_prefix_tail_block_indices_host.data() + r * MqaSharedAttentionMeta::kSwaSize;
    const auto& swa_block_ids = req->swa_kv_cache_blocks;
    const auto& swa_block_idx_offsets = req->swa_kv_cache_block_idx_offsets;
    const bool use_buffer = (req_buffer_start >= 0);
    int swa_start_blk = 0;
    int buffer_slots = 0;
    if (use_buffer) {
      const auto range = ComputeSwaCacheBufferRange(req->kv_cached_token_num, seq_len, mqa_shared.block_token_num);
      swa_start_blk = range.swa_start_blk;
      buffer_slots = range.slots;
    }
    for (int m = 0; m < kTailStride; ++m) {
      row[m] =
          use_buffer
              ? GetSwaBufferTokenPos(tail_first_j + m, mqa_shared.block_token_num, req_buffer_start, swa_start_blk)
              : GetSwaPoolTokenPos(swa_block_ids, swa_block_idx_offsets, tail_first_j + m, mqa_shared.block_token_num);
    }
    if (use_buffer) {
      req_buffer_start += buffer_slots;
    }
    ++r;
  }

  MemcpyAsync(mqa_shared.cu_extend_lens_tensor.GetPtr<void>(), mqa_shared.cu_extend_lens_host.data(),
              mqa_shared.cu_extend_lens_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
              config.context->device->GetH2DStream());
  MemcpyAsync(mqa_shared.cu_prefix_lens_tensor.GetPtr<void>(), mqa_shared.cu_prefix_lens_host.data(),
              mqa_shared.cu_prefix_lens_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
              config.context->device->GetH2DStream());
  MemcpyAsync(mqa_shared.swa_prefix_tail_block_indices_tensor.GetPtr<void>(),
              mqa_shared.swa_prefix_tail_block_indices_host.data(),
              mqa_shared.swa_prefix_tail_block_indices_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
              config.context->device->GetH2DStream());

  const int block_table_stride = static_cast<int>(standard_meta.block_table.shape[1]);
  const int aligned_c128_max =
      RoundUp(c128_max_seq_len_clamp1, static_cast<int>(MqaSharedAttentionMeta::kTopkBlockSize));

  // Slice per-meta views from the shared backing tensors.
  mqa_meta.swa_block_indices_tensor = mqa_shared.swa_block_indices_tensor.GetView(
      {total_dp_input_ids_len, MqaSharedAttentionMeta::kSwaSize}, mqa_shared.swa_block_indices_tensor.shape[0]);
  mqa_shared.swa_block_indices_tensor.shape[0] += mqa_meta.swa_block_indices_tensor.GetElementNumber();

  mqa_meta.c128_block_indices_tensor = mqa_shared.c128_block_indices_tensor.GetView(
      {total_dp_input_ids_len, static_cast<size_t>(aligned_c128_max)}, mqa_shared.c128_block_indices_tensor.shape[0]);
  mqa_shared.c128_block_indices_tensor.shape[0] += mqa_meta.c128_block_indices_tensor.GetElementNumber();

  mqa_meta.c128_topk_lengths_tensor =
      mqa_shared.c128_topk_lengths_tensor.GetView({batch_size}, mqa_shared.c128_topk_lengths_tensor.shape[0]);
  mqa_shared.c128_topk_lengths_tensor.shape[0] += mqa_meta.c128_topk_lengths_tensor.GetElementNumber();

  PrepareDsv4MqaMeta(
      mqa_shared.cu_extend_lens_tensor.GetPtr<const int32_t>(),
      mqa_shared.cu_prefix_lens_tensor.GetPtr<const int32_t>(), standard_meta.block_table.GetPtr<const int32_t>(),
      mqa_meta.raw_out_loc_tensor.GetPtr<const int32_t>(),
      mqa_shared.swa_prefix_tail_block_indices_tensor.GetPtr<const int32_t>(), batch_size, mqa_shared.block_token_num,
      block_table_stride, max_extend_len, mqa_meta.swa_block_indices_tensor.GetPtr<int32_t>(),
      mqa_meta.c128_block_indices_tensor.GetPtr<int32_t>(), mqa_meta.c128_topk_lengths_tensor.GetPtr<int32_t>(),
      MqaSharedAttentionMeta::kSwaSize, aligned_c128_max,
      /*fill_topk_with_stride=*/is_flash, config.context->device->GetH2DStream());
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

// Build pre/post D2D copy address lists for non-contiguous SWA requests.
// Buffer layout: slots cover FA block indices [swa_start_blk, end_blk] contiguously.
//   buffer slot for FA block g = req_buffer_start + (g - swa_start_blk)
// Pre-copy:  swa_block pool → buffer for any block whose start < kv_cached.
//   This includes the boundary block (global_blk == kv_cached_blk) when kv_cached is not
//   block-aligned: it already holds tokens [blk_start, kv_cached) from the pool.
// Post-copy: buffer → swa_block pool for FA blocks in [kv_cached_blk, end_blk].
//   The boundary block receives BOTH pre-copy and post-copy when kv_cached is unaligned.
// Only blocks with an actual physical SWA allocation (in swa_kv_cache_blocks) are copied;
// newly-inserted blocks in [kv_cached_blk, end_blk] are written by the KV insert kernel
// and then post-copied to the pool.
template <typename ReqPtrContainer>
void PrepareSwaCopyLists(MqaAttentionMeta& mqa_meta, MqaSharedAttentionMeta& mqa_shared, const ReqPtrContainer& dp_reqs,
                         const AttentionBuilderConfig& config) {
  const int block_token_num = mqa_shared.block_token_num;
  const size_t swa_block_size = static_cast<size_t>(mqa_shared.swa_block_size);
  // Each buffer slot stores one layer's SWA KV data (per_layer_swa_block_size bytes).
  // Layer offset is applied via BatchMemoryD2D src_offset/dst_offset at Forward time.
  const size_t per_layer = static_cast<size_t>(mqa_shared.per_layer_swa_block_size);

  int req_buffer_start = 0;
  for (const auto* req : dp_reqs) {
    const int kv_cached = req->kv_cached_token_num;
    const int seq_len = static_cast<int>(req->forwarding_tokens->size());
    // Buffer covers FA blocks [swa_start_blk, end_blk] — exactly the range needed by attention.
    const auto range = ComputeSwaCacheBufferRange(kv_cached, seq_len, block_token_num);

    const uint8_t* pool_base = static_cast<const uint8_t*>(req->swa_block_base_addr);
    const uint8_t* buf_base = mqa_shared.swa_cache_buffer.GetPtr<uint8_t>();
    const auto& offsets = req->swa_kv_cache_block_idx_offsets;
    const auto& blocks = req->swa_kv_cache_blocks;

    for (int i = 0; i < static_cast<int>(blocks.size()); ++i) {
      const int global_blk = static_cast<int>(offsets[i]);
      if (global_blk < range.swa_start_blk || global_blk > range.end_blk) continue;
      const int buffer_slot = req_buffer_start + (global_blk - range.swa_start_blk);
      const uint64_t pool_addr =
          reinterpret_cast<uint64_t>(pool_base + static_cast<size_t>(blocks[i]) * swa_block_size);
      const uint64_t buf_addr = reinterpret_cast<uint64_t>(buf_base + static_cast<size_t>(buffer_slot) * per_layer);

      // Pre-copy any block that already has cached tokens (pool → buffer).
      // Use global_blk * block_token_num < kv_cached rather than global_blk < kv_cached_blk
      // so the boundary block (kv_cached unaligned) is also pre-copied: it holds tokens
      // [blk_start, kv_cached) from the previous step and must be restored to the buffer.
      if (global_blk * block_token_num < kv_cached) {
        mqa_meta.swa_pre_copy_src_host.push_back(pool_addr);
        mqa_meta.swa_pre_copy_dst_host.push_back(buf_addr);
      }
      // Post-copy any block that will receive new KV data this step (buffer → pool).
      // The boundary block satisfies both conditions and gets pre- AND post-copied.
      if (global_blk >= range.kv_cached_blk) {
        mqa_meta.swa_post_copy_src_host.push_back(buf_addr);
        mqa_meta.swa_post_copy_dst_host.push_back(pool_addr);
      }
    }
    req_buffer_start += range.slots;
  }

  // H2D copy addr lists so that BatchMemoryD2D in Forward can use them directly.
  const size_t pre_n = mqa_meta.swa_pre_copy_src_host.size();
  if (pre_n > 0) {
    mqa_meta.swa_pre_copy_src_tensor = mqa_shared.swa_pre_copy_src_tensor.GetView({pre_n});
    mqa_meta.swa_pre_copy_dst_tensor = mqa_shared.swa_pre_copy_dst_tensor.GetView({pre_n});
    MemcpyAsync(mqa_meta.swa_pre_copy_src_tensor.GetPtr<void>(), mqa_meta.swa_pre_copy_src_host.data(),
                pre_n * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE, config.context->device->GetH2DStream());
    MemcpyAsync(mqa_meta.swa_pre_copy_dst_tensor.GetPtr<void>(), mqa_meta.swa_pre_copy_dst_host.data(),
                pre_n * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE, config.context->device->GetH2DStream());
  }
  const size_t post_n = mqa_meta.swa_post_copy_src_host.size();
  if (post_n > 0) {
    mqa_meta.swa_post_copy_src_tensor = mqa_shared.swa_post_copy_src_tensor.GetView({post_n});
    mqa_meta.swa_post_copy_dst_tensor = mqa_shared.swa_post_copy_dst_tensor.GetView({post_n});
    MemcpyAsync(mqa_meta.swa_post_copy_src_tensor.GetPtr<void>(), mqa_meta.swa_post_copy_src_host.data(),
                post_n * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE, config.context->device->GetH2DStream());
    MemcpyAsync(mqa_meta.swa_post_copy_dst_tensor.GetPtr<void>(), mqa_meta.swa_post_copy_dst_host.data(),
                post_n * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE, config.context->device->GetH2DStream());
  }
}

}  // namespace

// ── Public API ───────────────────────────────────────────────────────────────

void PrepareMqaFlash(MqaAttentionMeta& mqa_meta, MqaSharedAttentionMeta& mqa_shared,
                     const FlashAttentionMeta& flash_meta, const AttentionBuilderConfig& config) {
  if (flash_meta.dp_reqs.empty()) return;

  mqa_meta.Reset();

  InitSwaBasePointersIfNeeded(mqa_shared, flash_meta.dp_reqs);

  // Detect non-contiguous SWA, respecting RuntimeConfig override flags.
  const auto& rtcfg = config.runtime_config;
  if (!rtcfg.enable_non_contiguous_swa_mode) {
    mqa_meta.has_non_contiguous_swa = false;
  } else if (rtcfg.force_non_contiguous_swa_mode) {
    mqa_meta.has_non_contiguous_swa = true;
  } else {
    bool naturally_non_contiguous = false;
    for (const auto* req : flash_meta.dp_reqs) {
      const auto& offsets = req->swa_kv_cache_block_idx_offsets;
      for (size_t i = 0; i + 1 < offsets.size(); ++i) {
        if (offsets[i + 1] != offsets[i] + 1) {
          naturally_non_contiguous = true;
          break;
        }
      }
      if (naturally_non_contiguous) break;
    }
    mqa_meta.has_non_contiguous_swa = naturally_non_contiguous;
  }

  if (mqa_meta.has_non_contiguous_swa) {
    PrepareSwaCopyLists(mqa_meta, mqa_shared, flash_meta.dp_reqs, config);
  }

  PrepareOutLoc(mqa_meta, mqa_shared, flash_meta.dp_reqs, config);
  PrepareBlockIndicesAndTopkLengths(mqa_meta, mqa_shared, flash_meta.dp_reqs, flash_meta.total_dp_input_ids_len,
                                    static_cast<const StandardAttentionMeta&>(flash_meta), config,
                                    /*is_flash=*/true);

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
  PrepareBlockIndicesAndTopkLengths(mqa_meta, mqa_shared, paged_meta.dp_reqs, paged_meta.total_dp_input_ids_len,
                                    static_cast<const StandardAttentionMeta&>(paged_meta), config,
                                    /*is_flash=*/false);

  const size_t paged_batch = paged_meta.dp_reqs.size();
  AssignTileSchedulerAndNumSplits(mqa_meta, mqa_shared, paged_meta.q_seq_len, paged_batch, /*is_flash=*/false);
}

}  // namespace ksana_llm
