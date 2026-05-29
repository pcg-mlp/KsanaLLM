/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/attention_meta/mla_shared_attention_meta.h"

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/flash_mla/flash_mla.h"
#  include "csrc/kernels/nvidia/flash_mla/kernels/params.h"
#  include "csrc/kernels/nvidia/flash_sparse_mla/flash_sparse_mla.h"
#endif

namespace ksana_llm {

void MLASharedAttentionMeta::Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                  size_t decode_token_num_threshold, int rank) {
#ifdef ENABLE_CUDA
  if (!model_config.use_mla) {
    return;
  }

  const size_t max_batch_size = runtime_config.max_batch_size;
  const size_t head_num_per_tp = model_config.head_num / runtime_config.parallel_basic_config.attn_tensor_parallel_size;

  num_splits = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size + 1 + decode_token_num_threshold});

  for (size_t q_len = 1; q_len <= decode_token_num_threshold; ++q_len) {
    if (model_config.use_dsa) {
      tile_scheduler_metadatas.emplace_back(
          MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
          std::vector<size_t>{static_cast<size_t>(llm_kernels::nvidia::GetNumSmParts(head_num_per_tp, q_len)),
                              static_cast<size_t>(llm_kernels::nvidia::GetDecodingSchedMetaSize() / sizeof(int)), 1});
    } else {
      llm_kernels::nvidia::FlashMlaWorkspaceMap flash_mla_workspace_map;
      llm_kernels::nvidia::GetNumSmParts(flash_mla_workspace_map, q_len * head_num_per_tp, /*num_heads_k*/ 1, rank);
      tile_scheduler_metadatas.emplace_back(
          MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
          std::vector<size_t>{static_cast<size_t>(flash_mla_workspace_map.num_sm_parts),
                              llm_kernels::nvidia::TileSchedulerMetaDataSize});
    }
  }

  if (model_config.use_dsa) {
    tile_scheduler_metadatas.emplace_back(
        MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
        std::vector<size_t>{static_cast<size_t>(llm_kernels::nvidia::GetNumSmParts(head_num_per_tp, 1)),
                            static_cast<size_t>(llm_kernels::nvidia::GetDecodingSchedMetaSize() / sizeof(int)), 1});
  }

  // Chunked-KV buffers: pre-allocate to worst-case sizes (shapes overwritten per batch).
  max_chunked_kv_tokens = static_cast<int>(runtime_config.max_chunked_kv_tokens);
  chunk_kv_batch_starts_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {max_batch_size + 1});
  chunk_kv_cu_extend_lens_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {2 * max_batch_size});
  chunk_kv_cu_seq_lens_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {2 * max_batch_size});
#endif
}

}  // namespace ksana_llm
