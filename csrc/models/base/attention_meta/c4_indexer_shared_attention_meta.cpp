/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/attention_meta/c4_indexer_shared_attention_meta.h"

#include "configure/environment.h"

namespace ksana_llm {

void C4IndexerSharedAttentionMeta::Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                        int rank) {
  if (!model_config.use_dsv4_attn) {
    return;
  }

  // Allocate base tensors with C4 compress ratio (bypasses use_dsa guard in base Init).
  IndexerSharedAttentionMeta::InitCore(model_config, runtime_config, rank, /*compress_ratio_val=*/4);

  const size_t max_batch_size = runtime_config.max_batch_size;

  gathered_c4_out_loc_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {1});
  c4_seq_lens_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  extend_lens_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  // Always allocate: q_seq_len > 1 can occur regardless of GetDecodeTokenNumThreshold().
  expanded_c4_seq_lens_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size, GetDecodeTokenNumThreshold()});
}

}  // namespace ksana_llm
