/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/deepseek_v3/attention_meta/deepseek_v3_attention_meta.h"

namespace ksana_llm {

// Forward declaration.
class CompressorSlotResource;

// KimiK3 shared attention meta — inherits all DeepSeekV3 shared tensors.
struct KimiK3SharedAttentionMeta : public DeepSeekV3SharedAttentionMeta {
  // Pre-allocated shared backing tensors (per-batch views are sliced from these)
  Tensor state_indices_tensor;  // DEVICE, INT32, [max_batch_size]

  int layer_num_on_node = 0;
  bool layer_mamba_kv_cache_ptr_initialized = false;
  // layout: slot 0 = total_block_num (lazy), slot 1+2*i = conv ptr, slot 1+2*i+1 = recurrent ptr.
  // Per-layer values are offsets at Init, resolved to pointers at Prepare time.
  Tensor layer_mamba_kv_cache_ptr_tensor;  // HOST, INT64, [1 + 2 * layer_num_on_node]

  // A slot index s returned by the resource covers slots [s * decode_token_num, (s + 1) * decode_token_num).
  Tensor mamba_state_buffer;  // DEVICE, BYTES, [slot_pool_capacity, decode_token_num, block_cache_size]

  // Reuse slot resource from the deepseek-v4 compressor.
  CompressorSlotResource* slot_resource_ = nullptr;

  void Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int layer_num_on_node, int rank,
            std::shared_ptr<BaseModelCacheLayout> block_cache_layout);
};

struct KimiK3FlashAttentionMeta : public DeepSeekV3FlashAttentionMeta {
  // Mamba state indices: per-request mamba_kv_cache_block_id.
  Tensor state_indices_tensor;  // [batch], int32, device
  std::vector<int32_t> state_indices_host;

  // For causal conv1d.
  int max_extend_len_host{0};       // scalar, host — max(req.GetInputIdsLength()) across batch
  Tensor has_initial_state_tensor;  // [batch], bool, device — prefix_cache_len > 0 per request
  std::vector<uint8_t> has_initial_state_host;

  // For KDA chunked prefill (chunk_kda).
  Tensor chunk_indices_tensor;  // [num_chunks, 2], int32, device — (seq_idx, local_chunk_idx)
  Tensor chunk_offsets_tensor;  // [batch+1], int32, device — prefix-sum of ceil(seqlen/BT)
  std::vector<int32_t> chunk_indices_host;
  std::vector<int32_t> chunk_offsets_host;
  int num_chunks_host{0};  // scalar, host — total chunks across batch
};

struct KimiK3PagedAttentionMeta : public DeepSeekV3PagedAttentionMeta {
  // Mamba state indices: per-request mamba_kv_cache_block_id.
  Tensor state_indices_tensor;  // [batch], int32, device
  std::vector<int32_t> state_indices_host;
};

}  // namespace ksana_llm
