/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta.h"

#include <fmt/core.h>
#include <algorithm>

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/flash_sparse_mla/flash_sparse_mla.h"
#endif
#include "configure/environment.h"
#include "device/device_utils.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

void CompressorSharedAttentionMeta::Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                         int layer_num_on_node_param, int layer_start_on_node_param, int rank,
                                         std::shared_ptr<BaseModelCacheLayout> block_cache_layout) {
  if (!model_config.use_dsv4_attn) {
    return;
  }

  layer_num_on_node = layer_num_on_node_param;
  layer_start_on_node = layer_start_on_node_param;
  block_token_num = runtime_config.attn_backend_config.block_token_num;
  attn_dp_group_id = static_cast<uint32_t>(rank / runtime_config.parallel_basic_config.attn_tensor_parallel_size);
  attn_dp_rank_id = static_cast<uint32_t>(rank % runtime_config.parallel_basic_config.attn_tensor_parallel_size);

  const size_t max_batch_size = runtime_config.max_batch_size;
  const size_t max_token_num = runtime_config.max_step_token_num;
  // The slot pool must hold every concurrently in-flight request across all PP batches.
  // Under max_pp_batch_num > 1, the same CompressorSharedAttentionMeta is shared between PP
  // batches, and slots cannot be reclaimed until EvictFinishedRequests() is called.
  const size_t max_pp_batch_num = std::max<size_t>(runtime_config.max_pp_batch_num, 1);
  const size_t slot_pool_capacity = max_batch_size * max_pp_batch_num;

  c4_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  c128_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  // load_first_page | load_second_page | write_first_page | last_position
  c4_indices_bundle_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {4ul * max_batch_size});
  c128_load_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  c4_indices_prev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  seq_lens_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  extend_lens_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});

  c4_out_loc_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  c128_out_loc_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});

  if (runtime_config.enable_prefix_caching) {
    tail_read_block_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
    tail_target_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
    tail_write_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  }

  // kPrefillPlanDim from:
  // 3rdparty/LLM_kernels/csrc/kernels/nvidia/others/sglang/main/compressor/plan.h
  constexpr size_t kPrefillPlanDim = 16ul;
  c4_compress_plan_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES, {max_token_num, kPrefillPlanDim, 1});
  c4_write_plan_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES, {max_token_num, kPrefillPlanDim, 1});
  c128_compress_plan_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES, {max_token_num, kPrefillPlanDim, 1});
  c128_write_plan_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES, {max_token_num, kPrefillPlanDim, 1});

  layer_c4_state_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {static_cast<size_t>(layer_num_on_node)});
  layer_c4_indexer_state_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {static_cast<size_t>(layer_num_on_node)});
  layer_c128_state_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {static_cast<size_t>(layer_num_on_node)});

  // Pre-compute per-layer state base pointers from the cache layout.
  // The layout is fixed at startup; these pointers are reused across all forward steps.
  size_t* layer_c4_state_ptr = layer_c4_state_ptr_tensor.GetPtr<size_t>();
  size_t* layer_c4_indexer_state_ptr = layer_c4_indexer_state_ptr_tensor.GetPtr<size_t>();
  size_t* layer_c128_state_ptr = layer_c128_state_ptr_tensor.GetPtr<size_t>();

  const auto c4_state_layout = block_cache_layout->GetBlockCacheLayout("C4-state");
  const auto c4_indexer_state_layout = block_cache_layout->GetBlockCacheLayout("indexer-state");
  const auto c128_state_layout = block_cache_layout->GetBlockCacheLayout("C128-state");
  const auto& compress_ratios = model_config.dsv4_attn_config.compress_ratios;

  for (int layer_idx = 0; layer_idx < layer_num_on_node; ++layer_idx) {
    const auto c4_offsets = c4_state_layout->GetKvCacheLayerOffsets(layer_idx, "attention");
    KLLM_CHECK_WITH_INFO(c4_offsets.k_offset == c4_offsets.v_offset,
                         "CompressorSharedAttentionMeta requires k_offset == v_offset");
    const auto c4_indexer_offsets = c4_indexer_state_layout->GetKvCacheLayerOffsets(layer_idx, "attention");
    KLLM_CHECK_WITH_INFO(c4_indexer_offsets.k_offset == c4_indexer_offsets.v_offset,
                         "CompressorSharedAttentionMeta requires k_offset == v_offset");
    const auto c128_offsets = c128_state_layout->GetKvCacheLayerOffsets(layer_idx, "attention");
    KLLM_CHECK_WITH_INFO(c128_offsets.k_offset == c128_offsets.v_offset,
                         "CompressorSharedAttentionMeta requires k_offset == v_offset");

    *layer_c4_state_ptr++ = c4_offsets.k_offset;
    *layer_c4_indexer_state_ptr++ = c4_indexer_offsets.k_offset;
    *layer_c128_state_ptr++ = c128_offsets.k_offset;

    if (compress_ratios[layer_start_on_node + layer_idx] == 4) {
      ++c4_layer_num_on_node;
    } else if (compress_ratios[layer_start_on_node + layer_idx] == 128) {
      ++c128_layer_num_on_node;
    }
  }

  // When MTP is enabled, an extra slot is required to handle cross-boundary cases.
  if (GetDecodeTokenNumThreshold() > 1) {
    ++c4_slot_num;
    ++c128_slot_num;
  }

  c4_state_buffer =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES,
             {slot_pool_capacity, static_cast<size_t>(c4_slot_num), c4_state_layout->GetBlockCacheSize()});
  c4_indexer_state_buffer =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES,
             {slot_pool_capacity, static_cast<size_t>(c4_slot_num), c4_indexer_state_layout->GetBlockCacheSize()});
  c128_state_buffer =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BYTES,
             {slot_pool_capacity, static_cast<size_t>(c128_slot_num), c128_state_layout->GetBlockCacheSize()});

  KLLM_LOG_INFO << fmt::format(
      "CompressorSharedAttentionMeta init: c4_layer_num_on_node={} c4_slot_num={} "
      "c128_layer_num_on_node={} c128_slot_num={} "
      "slot_pool_capacity={} (= max_batch_size={} * max_pp_batch_num={})",
      c4_layer_num_on_node, c4_slot_num, c128_layer_num_on_node, c128_slot_num, slot_pool_capacity, max_batch_size,
      max_pp_batch_num);

  for (int i = 0; i < static_cast<int>(slot_pool_capacity); ++i) {
    free_slot.push(i);
  }
}

void MqaSharedAttentionMeta::Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                  int layer_num_on_node_param, int rank,
                                  std::shared_ptr<BaseModelCacheLayout> block_cache_layout) {
  if (!model_config.use_dsv4_attn) {
    return;
  }

  layer_num_on_node = layer_num_on_node_param;
  attn_dp_group_id = static_cast<uint32_t>(rank / runtime_config.parallel_basic_config.attn_tensor_parallel_size);
  attn_dp_rank_id = static_cast<uint32_t>(rank % runtime_config.parallel_basic_config.attn_tensor_parallel_size);
  block_size = runtime_config.attn_backend_config.block_size;
  block_token_num = runtime_config.attn_backend_config.block_token_num;
  index_topk = model_config.dsa_config.index_topk;

  KLLM_CHECK_WITH_INFO(block_token_num % static_cast<int>(kC4Ratio) == 0, "block_token_num must be a multiple of 4");
  KLLM_CHECK_WITH_INFO(block_token_num % static_cast<int>(kC128Ratio) == 0,
                       "block_token_num must be a multiple of 128");
  KLLM_CHECK_WITH_INFO(model_config.dsv4_attn_config.sliding_window == static_cast<int>(kSwaSize),
                       "sliding_window must be 128");
  KLLM_CHECK_WITH_INFO(index_topk % static_cast<int>(kTopkBlockSize) == 0,
                       "index_topk must be a multiple of kTopkBlockSize");

  // SWA-cache layout is unconditionally registered by DeepSeekV4ModelCacheLayout.
  KLLM_CHECK_WITH_INFO(block_cache_layout->IsBlockCacheTypeExist("SWA-cache"),
                       "MqaSharedAttentionMeta requires 'SWA-cache' layout to be registered.");
  swa_block_size = static_cast<int>(block_cache_layout->GetBlockCacheLayout("SWA-cache")->GetBlockCacheSize());
  KLLM_CHECK_WITH_INFO(swa_block_size > 0, "SWA-cache block size must be > 0");

  KLLM_LOG_INFO << fmt::format(
      "MqaSharedAttentionMeta init: rank={} layer_num_on_node={} block_size={} swa_block_size={}", rank,
      layer_num_on_node, block_size, swa_block_size);

  const size_t max_token_num = runtime_config.max_step_token_num;
  const size_t max_seq_len = runtime_config.max_seq_len;

  raw_out_loc_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  c128_topk_lengths_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  swa_block_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num, kSwaSize});
  c4_block_indices_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num, static_cast<size_t>(index_topk)});
  c128_block_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                                     {max_token_num, RoundUp(max_seq_len / kC128Ratio, kTopkBlockSize)});

  // layer_swa_kv_cache_ptr_tensor: slot 0 = device_block_num, slot 1 = swa_block_size, slot 2..N-1 = per-layer base
  layer_swa_kv_cache_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {2ul + static_cast<size_t>(layer_num_on_node)});
  // tail tensors: slot 0 = device_block_num, slot 1..N-1 = per-layer base
  layer_tail_attn_cache_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {1ul + static_cast<size_t>(layer_num_on_node)});
  layer_tail_indexer_cache_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {1ul + static_cast<size_t>(layer_num_on_node)});

  // Write swa_block_size into slot 1 once; it does not change across forward steps.
  *(layer_swa_kv_cache_ptr_tensor.GetPtr<int64_t>() + 1) = static_cast<int64_t>(swa_block_size);

  // Pre-compute per-layer SWA / tail cache offsets from the layout (fixed at startup).
  size_t* layer_swa_ptr = layer_swa_kv_cache_ptr_tensor.GetPtr<size_t>() + 2;
  size_t* layer_tail_attn_ptr = layer_tail_attn_cache_ptr_tensor.GetPtr<size_t>() + 1;
  size_t* layer_tail_indexer_ptr = layer_tail_indexer_cache_ptr_tensor.GetPtr<size_t>() + 1;
  const auto swa_block_layout = block_cache_layout->GetBlockCacheLayout("SWA-cache");
  for (int layer_idx = 0; layer_idx < layer_num_on_node; ++layer_idx) {
    const auto layer_offsets = swa_block_layout->GetKvCacheLayerOffsets(layer_idx, "swa");
    KLLM_CHECK_WITH_INFO(layer_offsets.k_offset == layer_offsets.v_offset,
                         "MqaSharedAttentionMeta requires k_offset == v_offset for SWA");
    *layer_swa_ptr++ = layer_offsets.k_offset;

    const auto tail_attn_offsets = swa_block_layout->GetKvCacheLayerOffsets(layer_idx, "tail_attn");
    *layer_tail_attn_ptr++ = tail_attn_offsets.k_offset;

    const auto tail_indexer_offsets = swa_block_layout->GetKvCacheLayerOffsets(layer_idx, "tail_indexer");
    *layer_tail_indexer_ptr++ = tail_indexer_offsets.k_offset;

    KLLM_LOG_DEBUG << fmt::format("layer {} swa_offset={} tail_attn_offset={} tail_indexer_offset={}", layer_idx,
                                  layer_offsets.k_offset, tail_attn_offsets.k_offset, tail_indexer_offsets.k_offset);
  }

#ifdef ENABLE_CUDA
  const size_t head_num_per_tp = model_config.head_num / runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  for (size_t q_len = 1; q_len <= GetDecodeTokenNumThreshold(); ++q_len) {
    c4_tile_scheduler_metadatas.emplace_back(
        MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
        std::vector<size_t>{static_cast<size_t>(llm_kernels::nvidia::GetNumSmParts(head_num_per_tp, q_len)),
                            static_cast<size_t>(llm_kernels::nvidia::GetDecodingSchedMetaSize() / sizeof(int)), 1});
    c128_tile_scheduler_metadatas.emplace_back(
        MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
        std::vector<size_t>{static_cast<size_t>(llm_kernels::nvidia::GetNumSmParts(head_num_per_tp, q_len)),
                            static_cast<size_t>(llm_kernels::nvidia::GetDecodingSchedMetaSize() / sizeof(int)), 1});
  }
  // Extra entry for flash (prefill): q_len=1 gives maximum num_sm_parts
  c4_tile_scheduler_metadatas.emplace_back(
      MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
      std::vector<size_t>{static_cast<size_t>(llm_kernels::nvidia::GetNumSmParts(head_num_per_tp, 1)),
                          static_cast<size_t>(llm_kernels::nvidia::GetDecodingSchedMetaSize() / sizeof(int)), 1});
  c128_tile_scheduler_metadatas.emplace_back(
      MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
      std::vector<size_t>{static_cast<size_t>(llm_kernels::nvidia::GetNumSmParts(head_num_per_tp, 1)),
                          static_cast<size_t>(llm_kernels::nvidia::GetDecodingSchedMetaSize() / sizeof(int)), 1});

  const size_t num_splits_size = runtime_config.max_batch_size + 1 + GetDecodeTokenNumThreshold();
  c4_num_splits_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {num_splits_size});
  c128_num_splits_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {num_splits_size});
#endif
}

}  // namespace ksana_llm
