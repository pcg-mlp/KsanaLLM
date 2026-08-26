/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/kimi_k3/attention_meta/kimi_k3_attention_meta.h"

#include <fmt/core.h>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/environment.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

void KimiK3SharedAttentionMeta::Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                     int layer_num_on_node_param, int rank,
                                     std::shared_ptr<BaseModelCacheLayout> block_cache_layout) {
  const size_t max_batch_size = runtime_config.max_batch_size;

  // Shared tensor for slicing.
  state_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});

  layer_num_on_node = layer_num_on_node_param;

  // slot 0 = total_block_num (lazy), slots 1..2N = per-layer conv/recurrent offsets.
  layer_mamba_kv_cache_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {1ul + 2ul * static_cast<size_t>(layer_num_on_node)});

  size_t* layer_mamba_ptr = layer_mamba_kv_cache_ptr_tensor.GetPtr<size_t>() + 1;
  const auto mamba_state_layout = block_cache_layout->GetBlockCacheLayout("mamba-state");
  for (int layer_idx = 0; layer_idx < layer_num_on_node; ++layer_idx) {
    const auto conv_offsets = mamba_state_layout->GetKvCacheLayerOffsets(layer_idx, "conv");
    KLLM_CHECK_WITH_INFO(conv_offsets.k_offset == conv_offsets.v_offset,
                         "KimiK3SharedAttentionMeta requires k_offset == v_offset for conv");
    const auto recurrent_offsets = mamba_state_layout->GetKvCacheLayerOffsets(layer_idx, "recurrent");
    KLLM_CHECK_WITH_INFO(recurrent_offsets.k_offset == recurrent_offsets.v_offset,
                         "KimiK3SharedAttentionMeta requires k_offset == v_offset for recurrent");
    *layer_mamba_ptr++ = conv_offsets.k_offset;
    *layer_mamba_ptr++ = recurrent_offsets.k_offset;

    KLLM_LOG_DEBUG << fmt::format("KimiK3 mamba layer {} conv_offset={} recurrent_offset={}", layer_idx,
                                  conv_offsets.k_offset, recurrent_offsets.k_offset);
  }

  // Mamba state buffer: per-request contiguous slot pool.
  // Total slots = slot_pool_capacity * decode_token_num.
  const size_t max_pp_batch_num = std::max<size_t>(runtime_config.max_pp_batch_num, 1);
  const size_t slot_pool_capacity = max_batch_size * max_pp_batch_num;
  const size_t decode_token_num = GetDecodeTokenNumThreshold();
  const size_t state_block_size = mamba_state_layout->GetBlockCacheSize();

  mamba_state_buffer =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES, {slot_pool_capacity, decode_token_num, state_block_size});

  KLLM_LOG_INFO << fmt::format(
      "KimiK3SharedAttentionMeta init: layer_num_on_node={} slot_pool_capacity={} "
      "(= max_batch_size={} * max_pp_batch_num={}) decode_token_num={} state_block_size={}",
      layer_num_on_node, slot_pool_capacity, max_batch_size, max_pp_batch_num, decode_token_num, state_block_size);
}

}  // namespace ksana_llm
