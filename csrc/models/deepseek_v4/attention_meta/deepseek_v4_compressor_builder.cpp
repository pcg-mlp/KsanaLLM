/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v4/attention_meta/deepseek_v4_compressor_builder.h"

#include <fmt/core.h>
#include <unordered_set>

#include "configure/environment.h"
#include "device/device_utils.h"
#include "models/base/forward_meta/common.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

namespace {

static constexpr int kC4Ratio = 4;
static constexpr int kC128Ratio = 128;

// Whether the request is handled by the paged (decode) path.
// Mirrors CompressorMeta::GetForwardRequestType(): q_seq_len==1 with cached tokens.
inline bool IsPaged(const ForwardRequest& req) { return req.GetInputIdsLength() == 1 && req.kv_cached_token_num > 0; }

// Translate a global block index to the physical SWA block ID (same as CompressorMeta::MapGlobalBlockToSwa).
int MapGlobalBlockToSwa(const std::vector<int>& swa_block_ids, const std::vector<size_t>& swa_block_idx_offsets,
                        int global_block_idx, const char* purpose) {
  const int swa_offset = swa_block_idx_offsets.empty() ? 0 : static_cast<int>(swa_block_idx_offsets[0]);
  const int local_idx = global_block_idx - swa_offset;
  KLLM_CHECK_WITH_INFO(
      local_idx >= 0 && local_idx < static_cast<int>(swa_block_ids.size()),
      fmt::format("compressor tail {}: global_block_idx {} -> local_swa_idx {} out of swa_kv_cache_blocks "
                  "(swa_offset={} swa_size={}).",
                  purpose, global_block_idx, local_idx, swa_offset, swa_block_ids.size()));
  return swa_block_ids[local_idx];
}

// Convert a logical token position to its physical cache position.
inline int GetPhysicalTokenPos(const std::vector<int32_t>& block_ids, int token_pos, int compress_ratio,
                               int block_token_num) {
  return token_pos < 0 ? -1
                       : block_ids[token_pos / block_token_num] * (block_token_num / compress_ratio) +
                             (token_pos / compress_ratio) % (block_token_num / compress_ratio);
}

inline int GetSubSlotId(int token_num, int compress_ratio, int c4_slot_num, int c128_slot_num) {
  if (compress_ratio == kC4Ratio) {
    return (token_num - 1) / kC4Ratio % c4_slot_num;
  } else {
    return (token_num - 1) / kC128Ratio % c128_slot_num;
  }
}

inline int GetC4PrevSlotId(int slot_id, int c4_slot_num) { return (slot_id + c4_slot_num - 1) % c4_slot_num; }

inline int GetPageIndex(int slot_id, int sub_slot, int compress_ratio, int c4_slot_num, int c128_slot_num,
                        int c4_layer_num_on_node, int c128_layer_num_on_node) {
  if (compress_ratio == kC4Ratio) {
    return (slot_id * c4_slot_num + sub_slot) * c4_layer_num_on_node;
  } else {
    return (slot_id * c128_slot_num + sub_slot) * c128_layer_num_on_node;
  }
}

void PreparePrefillPlan(const std::vector<const ForwardRequest*>& dp_reqs,
                        CompressorSharedAttentionMeta& compressor_shared) {
  size_t c4_compress_num = 0, c4_write_num = 0;
  size_t c128_compress_num = 0, c128_write_num = 0;
  for (const auto* req : dp_reqs) {
    const int extend_len = req->GetInputIdsLength();
    const int seq_len = static_cast<int>(req->forwarding_tokens->size());
    const int prefix_len = seq_len - extend_len;

    c4_compress_num += static_cast<size_t>(seq_len / kC4Ratio - prefix_len / kC4Ratio);
    {
      const int aligned = seq_len / kC4Ratio * kC4Ratio;
      const int start_write_pos = (aligned >= kC4Ratio) ? aligned - kC4Ratio : 0;
      if (start_write_pos <= prefix_len) {
        c4_write_num += static_cast<size_t>(extend_len);
      } else if (start_write_pos < seq_len) {
        c4_write_num += static_cast<size_t>(seq_len - start_write_pos);
      }
    }

    c128_compress_num += static_cast<size_t>(seq_len / kC128Ratio - prefix_len / kC128Ratio);
    {
      const int start_write_pos = seq_len / kC128Ratio * kC128Ratio;
      if (start_write_pos <= prefix_len) {
        c128_write_num += static_cast<size_t>(extend_len);
      } else if (start_write_pos < seq_len) {
        c128_write_num += static_cast<size_t>(seq_len - start_write_pos);
      }
    }
  }
  compressor_shared.c4_compress_plan_tensor.shape[0] = c4_compress_num;
  compressor_shared.c4_write_plan_tensor.shape[0] = c4_write_num;
  compressor_shared.c128_compress_plan_tensor.shape[0] = c128_compress_num;
  compressor_shared.c128_write_plan_tensor.shape[0] = c128_write_num;
}

void PrepareTailIndices(const std::vector<ForwardRequest>& reqs, const std::unordered_set<int64_t>& new_req_ids,
                        CompressorSharedAttentionMeta& compressor_shared, std::shared_ptr<Context> context) {
  if (compressor_shared.tail_read_block_tensor.shape.empty()) {
    return;
  }

  std::vector<int> tail_read_block_host;
  std::vector<int> tail_target_indices_host;

  for (const auto& req : reqs) {
    if (req.attn_dp_group_id != compressor_shared.attn_dp_group_id) continue;
    if (new_req_ids.find(req.req_id) == new_req_ids.end()) continue;

    const int seq_len = static_cast<int>(req.forwarding_tokens->size());
    const int extend_len = static_cast<int>(req.GetInputIdsLength());
    const int prefix_len = seq_len - extend_len;
    if (prefix_len > 0) {
      KLLM_CHECK_WITH_INFO(prefix_len % compressor_shared.block_token_num == 0,
                           "CompressorSharedAttentionMeta requires prefix_len to be a multiple of block_token_num");
      const int last_prefix_block_idx = (prefix_len - 1) / compressor_shared.block_token_num;
      tail_read_block_host.push_back(MapGlobalBlockToSwa(req.swa_kv_cache_blocks, req.swa_kv_cache_block_idx_offsets,
                                                         last_prefix_block_idx, "read"));
      const int slot_id = compressor_shared.req_to_slot.at(req.req_id);
      const int sub_slot_load =
          GetSubSlotId(prefix_len, kC4Ratio, compressor_shared.c4_slot_num, compressor_shared.c128_slot_num);
      tail_target_indices_host.push_back(
          GetPageIndex(slot_id, sub_slot_load, kC4Ratio, compressor_shared.c4_slot_num, compressor_shared.c128_slot_num,
                       compressor_shared.c4_layer_num_on_node, compressor_shared.c128_layer_num_on_node));
    }
  }

  auto& context_ = context;
  HOST_TO_DEVICE(tail_read_block_host, compressor_shared.tail_read_block_tensor);
  HOST_TO_DEVICE(tail_target_indices_host, compressor_shared.tail_target_indices_tensor);

  // ===== Tail-write indices (per-token) =====
  constexpr int kTailTokens = 4;
  const int tail_first_pos_in_block = compressor_shared.block_token_num - kTailTokens;
  std::vector<int> tail_write_indices_host;

  for (const auto& req : reqs) {
    if (req.attn_dp_group_id != compressor_shared.attn_dp_group_id) continue;

    const int seq_len = static_cast<int>(req.forwarding_tokens->size());
    const int extend_len = static_cast<int>(req.GetInputIdsLength());
    const int prefix_len = seq_len - extend_len;
    for (int i = prefix_len; i < seq_len; ++i) {
      const int pos_in_block = i % compressor_shared.block_token_num;
      if (pos_in_block >= tail_first_pos_in_block) {
        const int block_idx = i / compressor_shared.block_token_num;
        const int block_id =
            MapGlobalBlockToSwa(req.swa_kv_cache_blocks, req.swa_kv_cache_block_idx_offsets, block_idx, "write");
        const int slot = pos_in_block - tail_first_pos_in_block;
        tail_write_indices_host.push_back(block_id * kTailTokens | slot);
      } else {
        tail_write_indices_host.push_back(-1);
      }
    }
  }

  HOST_TO_DEVICE(tail_write_indices_host, compressor_shared.tail_write_indices_tensor);
}

void PrepareIndices(CompressorAttentionMeta& compressor, CompressorSharedAttentionMeta& compressor_shared,
                    bool is_paged, std::shared_ptr<Context> context) {
  // Local aliases so HOST_TO_SHARED_DEVICE macro resolves correctly.
  auto& context_ = context;
  auto& c4_indices_tensor = compressor_shared.c4_indices_tensor;
  auto& c128_indices_tensor = compressor_shared.c128_indices_tensor;
  auto& c4_indices_bundle_tensor = compressor_shared.c4_indices_bundle_tensor;
  auto& c128_load_indices_tensor = compressor_shared.c128_load_indices_tensor;
  auto& c4_indices_prev_tensor = compressor_shared.c4_indices_prev_tensor;

  const int c4_slot_num = compressor_shared.c4_slot_num;
  const int c128_slot_num = compressor_shared.c128_slot_num;
  const int c4_layer_num = compressor_shared.c4_layer_num_on_node;
  const int c128_layer_num = compressor_shared.c128_layer_num_on_node;

  for (const auto* req : compressor.dp_reqs) {
    const int seq_len = static_cast<int>(req->forwarding_tokens->size());
    const int extend_len = static_cast<int>(req->GetInputIdsLength());
    const int prefix_len = seq_len - extend_len;

    const int slot_id = compressor_shared.req_to_slot.at(req->req_id);
    const int c4_curr_slot_id = GetSubSlotId(seq_len, kC4Ratio, c4_slot_num, c128_slot_num);
    const int c4_prev_slot_id = GetC4PrevSlotId(c4_curr_slot_id, c4_slot_num);
    const int c4_curr_page =
        GetPageIndex(slot_id, c4_curr_slot_id, kC4Ratio, c4_slot_num, c128_slot_num, c4_layer_num, c128_layer_num);
    const int c4_prev_page =
        GetPageIndex(slot_id, c4_prev_slot_id, kC4Ratio, c4_slot_num, c128_slot_num, c4_layer_num, c128_layer_num);
    compressor.c4_indices_host.push_back(c4_curr_page);

    const int c128_curr_slot_id = GetSubSlotId(seq_len, kC128Ratio, c4_slot_num, c128_slot_num);
    const int c128_curr_page =
        GetPageIndex(slot_id, c128_curr_slot_id, kC128Ratio, c4_slot_num, c128_slot_num, c4_layer_num, c128_layer_num);
    compressor.c128_indices_host.push_back(c128_curr_page);

    if (!is_paged) {
      const int c4_load_second_slot_id = GetSubSlotId(prefix_len, kC4Ratio, c4_slot_num, c128_slot_num);
      const int c4_load_first_slot_id = GetC4PrevSlotId(c4_load_second_slot_id, c4_slot_num);
      const int load_second_page = GetPageIndex(slot_id, c4_load_second_slot_id, kC4Ratio, c4_slot_num, c128_slot_num,
                                                c4_layer_num, c128_layer_num);
      const int load_first_page = GetPageIndex(slot_id, c4_load_first_slot_id, kC4Ratio, c4_slot_num, c128_slot_num,
                                               c4_layer_num, c128_layer_num);
      compressor.c4_indices_bundle_host.push_back(load_first_page);
      compressor.c4_indices_bundle_host.push_back(load_second_page);
      compressor.c4_indices_bundle_host.push_back(c4_prev_page);
      const int last_position = (seq_len - 1) / kC4Ratio * kC4Ratio;
      compressor.c4_indices_bundle_host.push_back(last_position);
      const int c128_load_slot_id = GetSubSlotId(prefix_len, kC128Ratio, c4_slot_num, c128_slot_num);
      compressor.c128_load_indices_host.push_back(GetPageIndex(slot_id, c128_load_slot_id, kC128Ratio, c4_slot_num,
                                                               c128_slot_num, c4_layer_num, c128_layer_num));
    } else {
      compressor.c4_indices_prev_host.push_back(c4_prev_page);
    }
  }

  HOST_TO_SHARED_DEVICE(c4_indices, compressor);
  HOST_TO_SHARED_DEVICE(c128_indices, compressor);
  if (!is_paged) {
    HOST_TO_SHARED_DEVICE(c4_indices_bundle, compressor);
    HOST_TO_SHARED_DEVICE(c128_load_indices, compressor);
  } else {
    HOST_TO_SHARED_DEVICE(c4_indices_prev, compressor);
  }
}

void PrepareOutLoc(CompressorAttentionMeta& compressor, CompressorSharedAttentionMeta& compressor_shared,
                   std::shared_ptr<Context> context) {
  auto& context_ = context;
  auto& c4_out_loc_tensor = compressor_shared.c4_out_loc_tensor;
  auto& c128_out_loc_tensor = compressor_shared.c128_out_loc_tensor;

  for (const auto* req : compressor.dp_reqs) {
    const int extend_len = req->GetInputIdsLength();
    const int seq_len = static_cast<int>(req->forwarding_tokens->size());
    const int prefix_len = seq_len - extend_len;
    const auto& block_ids = req->atb_kv_cache_base_blk_ids[compressor_shared.attn_dp_rank_id];
    for (int i = prefix_len; i < seq_len; ++i) {
      if ((i + 1) % kC4Ratio == 0) {
        compressor.c4_out_loc_host.push_back(
            GetPhysicalTokenPos(block_ids, i, kC4Ratio, compressor_shared.block_token_num));
      } else {
        compressor.c4_out_loc_host.push_back(-1);
      }
      if ((i + 1) % kC128Ratio == 0) {
        compressor.c128_out_loc_host.push_back(
            GetPhysicalTokenPos(block_ids, i, kC128Ratio, compressor_shared.block_token_num));
      } else {
        compressor.c128_out_loc_host.push_back(-1);
      }
    }
  }
  HOST_TO_SHARED_DEVICE(c4_out_loc, compressor);
  HOST_TO_SHARED_DEVICE(c128_out_loc, compressor);
}

}  // namespace

void PrepareCompressor(const std::vector<ForwardRequest>& forward_reqs,
                       CompressorSharedAttentionMeta& compressor_shared, CompressorAttentionMeta& flash_compressor,
                       CompressorAttentionMeta& paged_compressor, std::shared_ptr<Context> context) {
  // Reset shared tensor offsets
  compressor_shared.c4_indices_tensor.shape = {0};
  compressor_shared.c128_indices_tensor.shape = {0};
  compressor_shared.c4_indices_bundle_tensor.shape = {0};
  compressor_shared.c128_load_indices_tensor.shape = {0};
  compressor_shared.c4_indices_prev_tensor.shape = {0};
  compressor_shared.seq_lens_tensor.shape = {0};
  compressor_shared.extend_lens_tensor.shape = {0};
  compressor_shared.c4_out_loc_tensor.shape = {0};
  compressor_shared.c128_out_loc_tensor.shape = {0};

  compressor_shared.c4_compress_plan_tensor.shape[2] = 0;
  compressor_shared.c4_write_plan_tensor.shape[2] = 0;
  compressor_shared.c128_compress_plan_tensor.shape[2] = 0;
  compressor_shared.c128_write_plan_tensor.shape[2] = 0;

  // Resolve state buffer base pointers once (block base is only known at runtime).
  if (__builtin_expect(!compressor_shared.layer_state_ptr_initialized, false)) {
    compressor_shared.layer_state_ptr_initialized = true;

    void** layer_c4_state_ptr = compressor_shared.layer_c4_state_ptr_tensor.GetPtr<void*>();
    void** layer_c4_indexer_state_ptr = compressor_shared.layer_c4_indexer_state_ptr_tensor.GetPtr<void*>();
    void** layer_c128_state_ptr = compressor_shared.layer_c128_state_ptr_tensor.GetPtr<void*>();
    uint8_t* const c4_state_ptr = compressor_shared.c4_state_buffer.GetPtr<uint8_t>();
    uint8_t* const c4_indexer_state_ptr = compressor_shared.c4_indexer_state_buffer.GetPtr<uint8_t>();
    uint8_t* const c128_state_ptr = compressor_shared.c128_state_buffer.GetPtr<uint8_t>();

    for (int layer_idx = 0; layer_idx < compressor_shared.layer_num_on_node; ++layer_idx) {
      *layer_c4_state_ptr = c4_state_ptr + reinterpret_cast<size_t>(*layer_c4_state_ptr);
      *layer_c4_indexer_state_ptr = c4_indexer_state_ptr + reinterpret_cast<size_t>(*layer_c4_indexer_state_ptr);
      *layer_c128_state_ptr = c128_state_ptr + reinterpret_cast<size_t>(*layer_c128_state_ptr);
      ++layer_c4_state_ptr;
      ++layer_c4_indexer_state_ptr;
      ++layer_c128_state_ptr;
    }
  }

  // Free slots for requests no longer in flight
  // NOTE: Slot eviction is intentionally NOT done here. Under max_pp_batch_num > 1, two PP batches
  // share this CompressorSharedAttentionMeta but each call only sees its own subset of running
  // requests, so evicting based on "not in current batch" would wrongly release slots that the
  // other PP batch is still using. Eviction is driven by EvictFinishedRequests(), invoked when
  // finish/recompute notifications arrive from the scheduler.

  // Allocate slots for new requests
  std::unordered_set<int64_t> new_req_ids;
  for (const auto& req : forward_reqs) {
    if (req.attn_dp_group_id != compressor_shared.attn_dp_group_id) continue;
    if (compressor_shared.req_to_slot.find(req.req_id) != compressor_shared.req_to_slot.end()) {
      continue;
    }
    KLLM_CHECK_WITH_INFO(
        !compressor_shared.free_slot.empty(),
        fmt::format("CompressorSharedAttentionMeta: free_slot pool exhausted (in-flight={}). "
                    "Total concurrent compressor requests across all PP batches exceeded the slot pool "
                    "size (max_batch_size * max_pp_batch_num). This usually means EvictFinishedRequests "
                    "was not called for some finished/recomputed requests.",
                    compressor_shared.req_to_slot.size()));
    compressor_shared.req_to_slot.emplace(req.req_id, compressor_shared.free_slot.front());
    compressor_shared.free_slot.pop();
    new_req_ids.insert(req.req_id);
  }

  PrepareTailIndices(forward_reqs, new_req_ids, compressor_shared, context);

  // Reset per-batch state
  flash_compressor.Reset();
  paged_compressor.Reset();

  // Categorise requests, accumulate dp_reqs / dp_tokens, and assert ordering.
  // Correctness of kv_score slicing in compressor.cpp depends on all !IsPaged requests
  // appearing before IsPaged requests within the same dp group (guaranteed by model_builder's
  // decode-descending sort: MTP decode token_num > 1 before std decode token_num == 1).
  bool seen_paged = false;
  for (const auto& req : forward_reqs) {
    if (req.attn_dp_group_id != compressor_shared.attn_dp_group_id) continue;
    if (!IsPaged(req)) {
      KLLM_CHECK_WITH_INFO(!seen_paged,
                           fmt::format("Compressor requires all flash (non-IsPaged) requests to appear before "
                                       "paged (IsPaged) requests in forward_reqs for a given dp group. "
                                       "Detected out-of-order request (req_id={}).",
                                       req.req_id));
      flash_compressor.dp_reqs.emplace_back(&req);
      flash_compressor.dp_tokens += req.GetInputIdsLength();
    } else {
      seen_paged = true;
      paged_compressor.dp_reqs.emplace_back(&req);
      paged_compressor.dp_tokens += req.GetInputIdsLength();
    }
  }

  auto& context_ = context;
  auto& seq_lens_tensor = compressor_shared.seq_lens_tensor;
  auto& extend_lens_tensor = compressor_shared.extend_lens_tensor;

  if (!flash_compressor.dp_reqs.empty()) {
    PrepareIndices(flash_compressor, compressor_shared, /*is_paged=*/false, context);
    PrepareOutLoc(flash_compressor, compressor_shared, context);
    for (const auto* req : flash_compressor.dp_reqs) {
      flash_compressor.seq_lens_host.push_back(static_cast<int>(req->forwarding_tokens->size()));
      flash_compressor.extend_lens_host.push_back(static_cast<int>(req->GetInputIdsLength()));
    }
    HOST_TO_SHARED_DEVICE(seq_lens, flash_compressor);
    HOST_TO_SHARED_DEVICE(extend_lens, flash_compressor);
    PreparePrefillPlan(flash_compressor.dp_reqs, compressor_shared);
  }

  if (!paged_compressor.dp_reqs.empty()) {
    PrepareIndices(paged_compressor, compressor_shared, /*is_paged=*/true, context);
    PrepareOutLoc(paged_compressor, compressor_shared, context);
    for (const auto* req : paged_compressor.dp_reqs) {
      paged_compressor.seq_lens_host.push_back(static_cast<int>(req->forwarding_tokens->size()));
    }
    HOST_TO_SHARED_DEVICE(seq_lens, paged_compressor);
  }
}

void EvictFinishedRequests(const std::vector<int64_t>& finished_req_ids,
                           CompressorSharedAttentionMeta& compressor_shared) {
  for (int64_t req_id : finished_req_ids) {
    auto it = compressor_shared.req_to_slot.find(req_id);
    if (it == compressor_shared.req_to_slot.end()) {
      continue;
    }
    compressor_shared.free_slot.push(it->second);
    compressor_shared.req_to_slot.erase(it);
  }
}

}  // namespace ksana_llm
