/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/attention_meta/indexer_shared_attention_meta.h"

#include "configure/environment.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

void IndexerSharedAttentionMeta::Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                      int rank) {
  if (!model_config.use_dsa) {
    return;
  }
  InitCore(model_config, runtime_config, rank, /*compress_ratio_val=*/1);
}

void IndexerSharedAttentionMeta::InitCore(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                          int rank, int compress_ratio_val) {
  compress_ratio = compress_ratio_val;
  attn_dp_group_id =
      static_cast<uint32_t>(rank / runtime_config.parallel_basic_config.attn_tensor_parallel_size);
  max_chunked_q_tokens = static_cast<int>(
      ComputeIndexerMaxQChunkSize(runtime_config.max_seq_len / compress_ratio_val, runtime_config.max_step_token_num));
  max_chunked_kv_tokens = static_cast<int>(runtime_config.max_chunked_kv_tokens);

  const size_t max_batch_size = runtime_config.max_batch_size;
  const size_t max_token_num = runtime_config.max_step_token_num;

  chunk_token_starts_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {max_batch_size + 1});
  chunk_kv_token_ranges_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {max_batch_size * 2});

  cu_seq_len_start_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  cu_seq_len_end_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  batch_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  // Always allocate: q_seq_len > 1 can occur regardless of GetDecodeTokenNumThreshold().
  expanded_seq_len_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size, GetDecodeTokenNumThreshold()});
}

}  // namespace ksana_llm
