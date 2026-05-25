/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/base/forward_meta/compressor_meta.h"

#include <fmt/core.h>
#include <unordered_set>

#include "configure/environment.h"
#include "device/device_utils.h"
#include "models/base/forward_meta/common.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

CompressorMeta::CompressorMeta(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                               int layer_num_on_node, int layer_start_on_node, int rank,
                               std::shared_ptr<Context> context,
                               std::shared_ptr<BaseModelCacheLayout> block_cache_layout)
    : layer_num_on_node_(layer_num_on_node), layer_start_on_node_(layer_start_on_node), context_(context) {
  attn_dp_group_id_ = rank / runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  attn_dp_rank_id_ = rank % runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  block_token_num_ = runtime_config.attn_backend_config.block_token_num;

  const size_t max_batch_size = runtime_config.max_batch_size;
  c4_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  c128_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  c4_indices_bundle_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
             {4ul * max_batch_size});  // load_first_page | load_second_page | write_first_page | last_position
  c128_load_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  c4_indices_prev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  seq_lens_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  extend_lens_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});

  const size_t max_token_num = runtime_config.max_step_token_num;
  c4_out_loc_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  c128_out_loc_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  if (runtime_config.enable_prefix_caching) {
    // Tail indices for prefix caching: read is per-request, write is per_token.
    tail_read_block_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
    tail_target_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
    tail_write_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  }
  constexpr size_t kPrefillPlanDim =
      16ul;  // from 3rdparty/LLM_kernels/csrc/kernels/nvidia/others/sglang/main/compressor/plan.h
             // Used by flash compress prefill
  c4_compress_plan_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES, {max_token_num, kPrefillPlanDim, 1});
  c4_write_plan_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES, {max_token_num, kPrefillPlanDim, 1});
  c128_compress_plan_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES, {max_token_num, kPrefillPlanDim, 1});
  c128_write_plan_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES, {max_token_num, kPrefillPlanDim, 1});

  layer_c4_state_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {static_cast<size_t>(layer_num_on_node_)});
  layer_c4_indexer_state_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {static_cast<size_t>(layer_num_on_node_)});
  layer_c128_state_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {static_cast<size_t>(layer_num_on_node_)});
  // Pre-compute kv cache types and per-layer offsets once. The layout is fixed at startup,
  // so subsequent forward steps can reuse the cached results instead of recomputing them.
  size_t* layer_c4_state_ptr = layer_c4_state_ptr_tensor.GetPtr<size_t>();
  size_t* layer_c4_indexer_state_ptr = layer_c4_indexer_state_ptr_tensor.GetPtr<size_t>();
  size_t* layer_c128_state_ptr = layer_c128_state_ptr_tensor.GetPtr<size_t>();
  const auto c4_state_layout = block_cache_layout->GetBlockCacheLayout("C4-state");
  const auto c4_indexer_state_layout = block_cache_layout->GetBlockCacheLayout("indexer-state");
  const auto c128_state_layout = block_cache_layout->GetBlockCacheLayout("C128-state");
  const auto& compress_ratios = model_config.dsv4_attn_config.compress_ratios;
  for (int layer_idx = 0; layer_idx < layer_num_on_node_; ++layer_idx) {
    const auto c4_layer_offsets = c4_state_layout->GetKvCacheLayerOffsets(layer_idx, /*kv_cache_type*/ "attention");
    KLLM_CHECK_WITH_INFO(c4_layer_offsets.k_offset == c4_layer_offsets.v_offset,
                         "CompressorMeta requires k_offset == v_offset");
    const auto c4_indexer_layer_offsets =
        c4_indexer_state_layout->GetKvCacheLayerOffsets(layer_idx, /*kv_cache_type*/ "attention");
    KLLM_CHECK_WITH_INFO(c4_indexer_layer_offsets.k_offset == c4_indexer_layer_offsets.v_offset,
                         "CompressorMeta requires k_offset == v_offset");
    const auto c128_layer_offsets = c128_state_layout->GetKvCacheLayerOffsets(layer_idx, /*kv_cache_type*/ "attention");
    KLLM_CHECK_WITH_INFO(c128_layer_offsets.k_offset == c128_layer_offsets.v_offset,
                         "CompressorMeta requires k_offset == v_offset");
    // Record offsets first
    *layer_c4_state_ptr++ = c4_layer_offsets.k_offset;
    *layer_c4_indexer_state_ptr++ = c4_indexer_layer_offsets.k_offset;
    *layer_c128_state_ptr++ = c128_layer_offsets.k_offset;

    if (compress_ratios[layer_start_on_node_ + layer_idx] == 4) {
      ++c4_layer_num_on_node_;
    } else if (compress_ratios[layer_start_on_node_ + layer_idx] == 128) {
      ++c128_layer_num_on_node_;
    }
  }

  // When MTP is enabled, an additional slot is required to handle cross-boundary cases
  if (GetDecodeTokenNumThreshold() > 1) {
    ++c4_slot_num;
    ++c128_slot_num;
  }
  c4_state_buffer = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES,
                           {max_batch_size, static_cast<size_t>(c4_slot_num), c4_state_layout->GetBlockCacheSize()});
  c4_indexer_state_buffer =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES,
             {max_batch_size, static_cast<size_t>(c4_slot_num), c4_indexer_state_layout->GetBlockCacheSize()});
  c128_state_buffer =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES,
             {max_batch_size, static_cast<size_t>(c128_slot_num), c128_state_layout->GetBlockCacheSize()});
  KLLM_LOG_INFO << fmt::format(
      "CompressorMeta init: c4_layer_num_on_node={} c4_slot_num={} c128_layer_num_on_node={} c128_slot_num={}",
      c4_layer_num_on_node_, c4_slot_num, c128_layer_num_on_node_, c128_slot_num);

  for (int i = 0; i < static_cast<int>(max_batch_size); ++i) {
    free_slot_.push(i);
  }
}

void CompressorMeta::Prepare(const std::vector<ForwardRequest>& forward_reqs) {
  // Reset the offsets of shared tensors
  c4_indices_tensor.shape = {0};
  c128_indices_tensor.shape = {0};
  c4_indices_bundle_tensor.shape = {0};
  c128_load_indices_tensor.shape = {0};
  c4_indices_prev_tensor.shape = {0};
  seq_lens_tensor.shape = {0};
  extend_lens_tensor.shape = {0};
  c4_out_loc_tensor.shape = {0};
  c128_out_loc_tensor.shape = {0};

  c4_compress_plan_tensor.shape[2] = 0;
  c4_write_plan_tensor.shape[2] = 0;
  c128_compress_plan_tensor.shape[2] = 0;
  c128_write_plan_tensor.shape[2] = 0;

  // Block base pointer is only known at runtime.
  if (__builtin_expect(!layer_state_ptr_initialized_, false)) {  // unlikely
    layer_state_ptr_initialized_ = true;

    void** layer_c4_state_ptr = layer_c4_state_ptr_tensor.GetPtr<void*>();
    void** layer_c4_indexer_state_ptr = layer_c4_indexer_state_ptr_tensor.GetPtr<void*>();
    void** layer_c128_state_ptr = layer_c128_state_ptr_tensor.GetPtr<void*>();
    uint8_t* const c4_state_ptr = c4_state_buffer.GetPtr<uint8_t>();
    uint8_t* const c4_indexer_state_ptr = c4_indexer_state_buffer.GetPtr<uint8_t>();
    uint8_t* const c128_state_ptr = c128_state_buffer.GetPtr<uint8_t>();
    for (int layer_idx = 0; layer_idx < layer_num_on_node_; ++layer_idx) {
      *layer_c4_state_ptr = c4_state_ptr + reinterpret_cast<size_t>(*layer_c4_state_ptr);
      *layer_c4_indexer_state_ptr = c4_indexer_state_ptr + reinterpret_cast<size_t>(*layer_c4_indexer_state_ptr);
      *layer_c128_state_ptr = c128_state_ptr + reinterpret_cast<size_t>(*layer_c128_state_ptr);
      ++layer_c4_state_ptr;
      ++layer_c4_indexer_state_ptr;
      ++layer_c128_state_ptr;
    }
  }

  // Free unused slots first
  std::unordered_set<int64_t> inflight_reqs;
  for (const auto& req : forward_reqs) {
    if (req.attn_dp_group_id == attn_dp_group_id_) {
      inflight_reqs.insert(req.req_id);
    }
  }
  for (auto it = req_to_slot_.begin(); it != req_to_slot_.end();) {
    if (inflight_reqs.find(it->first) == inflight_reqs.end()) {
      free_slot_.push(it->second);
      it = req_to_slot_.erase(it);
    } else {
      ++it;
    }
  }

  // Allocate slots for new requests
  std::unordered_set<int64_t> new_req_ids;
  for (const auto& req : forward_reqs) {
    if (req.attn_dp_group_id != attn_dp_group_id_) {
      continue;
    }
    if (req_to_slot_.find(req.req_id) == req_to_slot_.end()) {
      KLLM_CHECK_WITH_INFO(!free_slot_.empty(), "free_slot should not be empty");
      req_to_slot_.emplace(req.req_id, free_slot_.front());
      free_slot_.pop();
      new_req_ids.insert(req.req_id);
    }
  }

  PrepareTailIndices(forward_reqs, new_req_ids);

  // Reset meta
  flash_input.Reset();
  page_input.Reset();
  dp_context_tokens = 0;
  dp_decode_tokens = 0;

  // Categorize requests
  for (const auto& req : forward_reqs) {
    if (req.attn_dp_group_id == attn_dp_group_id_) {
      if (GetForwardRequestType(req) == ForwardRequestType::kFlash) {
        flash_input.dp_reqs.emplace_back(&req);
        dp_context_tokens += req.GetInputIdsLength();
      } else {  // GetForwardRequestType(req) == ForwardRequestType::kPage
        page_input.dp_reqs.emplace_back(&req);
        dp_decode_tokens += req.GetInputIdsLength();
      }
    }
  }

  if (!flash_input.dp_reqs.empty()) {
    PrepareIndices(flash_input);
    PrepareOutLoc(flash_input);
    for (const auto* req : flash_input.dp_reqs) {
      PrepareSeqLens(flash_input.seq_lens_host, *req);
      PrepareExtendLens(flash_input.extend_lens_host, *req);
    }
    HOST_TO_SHARED_DEVICE(seq_lens, flash_input);
    HOST_TO_SHARED_DEVICE(extend_lens, flash_input);

    // Compute exact compress_num and write_num for plan tensors
    PreparePrefillPlan(flash_input);
  }
  if (!page_input.dp_reqs.empty()) {
    PrepareIndices(page_input, /*is_paged*/ true);
    PrepareOutLoc(page_input);
    for (const auto* req : page_input.dp_reqs) {
      PrepareSeqLens(page_input.seq_lens_host, *req);
    }
    HOST_TO_SHARED_DEVICE(seq_lens, page_input);
  }
}

void CompressorMeta::PreparePrefillPlan(const input_info& input) {
  size_t c4_compress_num = 0, c4_write_num = 0;
  size_t c128_compress_num = 0, c128_write_num = 0;
  for (const auto* req : input.dp_reqs) {
    const int extend_len = req->GetInputIdsLength();
    const int seq_len = req->forwarding_tokens->size();
    const int prefix_len = seq_len - extend_len;

    // C4 (overlap): compress entries are positions where (pos+1) % 4 == 0
    c4_compress_num += static_cast<size_t>(seq_len / kC4Ratio - prefix_len / kC4Ratio);
    // C4 write: start_write_pos = aligned_down(seq_len, 4) - 4 (clamped to 0)
    {
      const int aligned = seq_len / kC4Ratio * kC4Ratio;
      const int start_write_pos = (aligned >= kC4Ratio) ? aligned - kC4Ratio : 0;
      if (start_write_pos <= prefix_len) {
        c4_write_num += static_cast<size_t>(extend_len);
      } else if (start_write_pos < seq_len) {
        c4_write_num += static_cast<size_t>(seq_len - start_write_pos);
      }
    }

    // C128 (non-overlap): compress entries are positions where (pos+1) % 128 == 0
    c128_compress_num += static_cast<size_t>(seq_len / kC128Ratio - prefix_len / kC128Ratio);
    // C128 write: start_write_pos = aligned_down(seq_len, 128)
    {
      const int start_write_pos = seq_len / kC128Ratio * kC128Ratio;
      if (start_write_pos <= prefix_len) {
        c128_write_num += static_cast<size_t>(extend_len);
      } else if (start_write_pos < seq_len) {
        c128_write_num += static_cast<size_t>(seq_len - start_write_pos);
      }
    }
  }
  c4_compress_plan_tensor.shape[0] = c4_compress_num;
  c4_write_plan_tensor.shape[0] = c4_write_num;
  c128_compress_plan_tensor.shape[0] = c128_compress_num;
  c128_write_plan_tensor.shape[0] = c128_write_num;
}

void CompressorMeta::PrepareTailIndices(const std::vector<ForwardRequest>& reqs,
                                        const std::unordered_set<int64_t>& new_req_ids) {
  if (tail_read_block_tensor.shape.empty()) {
    return;
  }

  // ===== Tail-read 索引 (per-request, 仅新请求需要) =====
  tail_read_block_host.clear();
  tail_target_indices_host.clear();

  for (const auto& req : reqs) {
    if (req.attn_dp_group_id != attn_dp_group_id_) {
      continue;
    }

    if (new_req_ids.find(req.req_id) == new_req_ids.end()) {
      continue;
    }

    const int seq_len = static_cast<int>(req.forwarding_tokens->size());
    const int extend_len = static_cast<int>(req.GetInputIdsLength());
    const int prefix_len = seq_len - extend_len;
    if (prefix_len > 0) {
      KLLM_CHECK_WITH_INFO(prefix_len % block_token_num_ == 0,
                           "CompressorMeta requires prefix_len to be a multiple of block_token_num");
      const int last_prefix_block_idx = (prefix_len - 1) / block_token_num_;
      tail_read_block_host.push_back(MapGlobalBlockToSwa(req.swa_kv_cache_blocks, req.swa_kv_cache_block_idx_offsets,
                                                         last_prefix_block_idx, "read"));
      const int slot_id = req_to_slot_.at(req.req_id);
      const int sub_slot_load = GetSubSlotId(prefix_len, kC4Ratio);
      tail_target_indices_host.push_back(GetPageIndex(slot_id, sub_slot_load, kC4Ratio));
    }
  }

  // Upload tail-read tensors
  HOST_TO_DEVICE(tail_read_block_host, tail_read_block_tensor);
  HOST_TO_DEVICE(tail_target_indices_host, tail_target_indices_tensor);

  // ===== Tail-write 索引 (per-token) =====
  // 每个 KV block 内最后 kTailTokens 个 token 是被持久化到 tail 区的"源 token"。
  // 槽位映射: pos_in_block ∈ [BTN-4, BTN-1]  →  tail_slot ∈ [0, 3]
  constexpr int kTailTokens = 4;
  const int tail_first_pos_in_block = block_token_num_ - kTailTokens;

  tail_write_indices_host.clear();

  // tail_write_indices[i]: packed encoding of (block_id, tail_slot) for the i-th token.
  //   Encoding: block_id * kTailTokens | tail_slot  (tail_slot ∈ [0,3], uses lower 2 bits)
  //   Value -1 means the token is not in a tail position; kernel early-exits.
  for (const auto& req : reqs) {
    if (req.attn_dp_group_id != attn_dp_group_id_) {
      continue;
    }

    const int seq_len = static_cast<int>(req.forwarding_tokens->size());
    const int extend_len = static_cast<int>(req.GetInputIdsLength());
    const int prefix_len = seq_len - extend_len;
    for (int i = prefix_len; i < seq_len; ++i) {
      const int pos_in_block = i % block_token_num_;
      if (pos_in_block >= tail_first_pos_in_block) {
        const int block_idx = i / block_token_num_;
        const int block_id =
            MapGlobalBlockToSwa(req.swa_kv_cache_blocks, req.swa_kv_cache_block_idx_offsets, block_idx, "write");
        const int slot = pos_in_block - tail_first_pos_in_block;
        tail_write_indices_host.push_back(block_id * kTailTokens | slot);
      } else {
        tail_write_indices_host.push_back(-1);
      }
    }
  }

  // Upload tail-write tensors
  HOST_TO_DEVICE(tail_write_indices_host, tail_write_indices_tensor);
}

void CompressorMeta::PrepareIndices(input_info& input, bool is_paged) {
  const auto& reqs = input.dp_reqs;
  const int batch_size = static_cast<int>(reqs.size());

  // 索引语义:
  //   slot_id    — free_slot_ 分配，req_to_slot_ 持有。
  //   sub_slot   — GetSubSlotId(token_num, ratio)；prev_slot — GetC4PrevSlotId(sub_slot)。
  //   page_index — GetPageIndex(slot_id, sub_slot, ratio)，定位 state_buffer 中的页。
  //   bundle (prefill only) = {load_first_page, load_second_page, write_first_page, last_position}。

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const auto& req = *reqs[batch_id];
    const int seq_len = static_cast<int>(req.forwarding_tokens->size());
    const int extend_len = static_cast<int>(req.GetInputIdsLength());
    const int prefix_len = seq_len - extend_len;

    const int slot_id = req_to_slot_.at(req.req_id);
    const int c4_curr_slot_id = GetSubSlotId(seq_len, kC4Ratio);
    const int c4_prev_slot_id = GetC4PrevSlotId(c4_curr_slot_id);
    const int c4_curr_page = GetPageIndex(slot_id, c4_curr_slot_id, kC4Ratio);
    const int c4_prev_page = GetPageIndex(slot_id, c4_prev_slot_id, kC4Ratio);
    input.c4_indices_host.push_back(c4_curr_page);
    const int c128_curr_slot_id = GetSubSlotId(seq_len, kC128Ratio);
    const int c128_curr_page = GetPageIndex(slot_id, c128_curr_slot_id, kC128Ratio);
    input.c128_indices_host.push_back(c128_curr_page);

    if (!is_paged) {
      const int c4_load_second_slot_id = GetSubSlotId(prefix_len, kC4Ratio);
      const int c4_load_first_slot_id = GetC4PrevSlotId(c4_load_second_slot_id);
      const int load_second_page = GetPageIndex(slot_id, c4_load_second_slot_id, kC4Ratio);
      const int load_first_page = GetPageIndex(slot_id, c4_load_first_slot_id, kC4Ratio);
      input.c4_indices_bundle_host.push_back(load_first_page);
      input.c4_indices_bundle_host.push_back(load_second_page);
      input.c4_indices_bundle_host.push_back(c4_prev_page);  // write_first_page
      const int last_position = (seq_len - 1) / kC4Ratio * kC4Ratio;
      input.c4_indices_bundle_host.push_back(last_position);  // last_position is a token offset, not a page index
      const int c128_load_slot_id = GetSubSlotId(prefix_len, kC128Ratio);
      input.c128_load_indices_host.push_back(GetPageIndex(slot_id, c128_load_slot_id, kC128Ratio));
    } else {
      // Decode: 仅需要记录上一个 c4 group 所在 page。
      input.c4_indices_prev_host.push_back(c4_prev_page);
    }
  }

  HOST_TO_SHARED_DEVICE(c4_indices, input);
  HOST_TO_SHARED_DEVICE(c128_indices, input);
  if (!is_paged) {
    HOST_TO_SHARED_DEVICE(c4_indices_bundle, input);
    HOST_TO_SHARED_DEVICE(c128_load_indices, input);
  } else {
    HOST_TO_SHARED_DEVICE(c4_indices_prev, input);
  }
}

int CompressorMeta::MapGlobalBlockToSwa(const std::vector<int>& swa_block_ids,
                                        const std::vector<size_t>& swa_block_idx_offsets, int global_block_idx,
                                        const char* purpose) {
  // 通过 swa_kv_cache_blocks + swa_kv_cache_block_idx_offsets 单独映射
  // NOTE: 当前底层算子尚未支持不连续 SWA block, 此处仍假定 swa_kv_cache_block_idx_offsets 内的索引连续,
  //       使用 swa_kv_cache_block_idx_offsets[0] 作为起始偏移做算术映射
  const int swa_offset = swa_block_idx_offsets.empty() ? 0 : static_cast<int>(swa_block_idx_offsets[0]);
  const int local_idx = global_block_idx - swa_offset;
  KLLM_CHECK_WITH_INFO(
      local_idx >= 0 && local_idx < static_cast<int>(swa_block_ids.size()),
      fmt::format("compressor tail {}: global_block_idx {} -> local_swa_idx {} out of swa_kv_cache_blocks "
                  "(swa_offset={} swa_size={}).",
                  purpose, global_block_idx, local_idx, swa_offset, swa_block_ids.size()));
  return swa_block_ids[local_idx];
}

void CompressorMeta::PrepareOutLoc(input_info& input) {
  for (const auto* req : input.dp_reqs) {
    const int extend_len = req->GetInputIdsLength();
    const int seq_len = req->forwarding_tokens->size();
    const int prefix_len = seq_len - extend_len;
    const auto& block_ids = req->atb_kv_cache_base_blk_ids[attn_dp_rank_id_];
    for (int i = prefix_len; i < seq_len; ++i) {
      // c4_out_loc: valid at multiples of C4 ratio
      if ((i + 1) % kC4Ratio == 0) {
        input.c4_out_loc_host.push_back(GetPhysicalTokenPos(block_ids, i, kC4Ratio));
      } else {
        input.c4_out_loc_host.push_back(-1);
      }
      // c128_out_loc: valid at multiples of C128 ratio
      if ((i + 1) % kC128Ratio == 0) {
        input.c128_out_loc_host.push_back(GetPhysicalTokenPos(block_ids, i, kC128Ratio));
      } else {
        input.c128_out_loc_host.push_back(-1);
      }
    }
  }
  HOST_TO_SHARED_DEVICE(c4_out_loc, input);
  HOST_TO_SHARED_DEVICE(c128_out_loc, input);
}

void CompressorMeta::PrepareSeqLens(std::vector<int>& host, const ForwardRequest& req) {
  host.push_back(static_cast<int>(req.forwarding_tokens->size()));
}

void CompressorMeta::PrepareExtendLens(std::vector<int>& host, const ForwardRequest& req) {
  host.push_back(static_cast<int>(req.GetInputIdsLength()));
}

}  // namespace ksana_llm
