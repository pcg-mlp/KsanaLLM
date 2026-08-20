/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/attention_meta/mla_attention_meta_builder.h"

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/flash_mla/flash_mla.h"
#endif
#include "configure/environment.h"
#include "profiler/profile_event.h"

namespace ksana_llm {

void PrepareFlashMla(MLAAttentionMeta& mla_meta, DSAAttentionMeta& dsa_meta, MLASharedAttentionMeta& mla_shared,
                     DSASharedAttentionMeta& dsa_shared, const StandardAttentionMeta& input,
                     const AttentionBuilderConfig& config) {
#ifdef ENABLE_CUDA
  if (input.dp_reqs.empty()) {
    return;
  }
  PROFILE_EVENT_SCOPE(PrepareFlashMla, "PrepareFlashMla", config.rank);
  const size_t batch_size = input.q_seq_len == 0 ? 1 : input.dp_reqs.size();
  mla_meta.tile_scheduler_metadata =
      mla_shared.tile_scheduler_metadatas[input.q_seq_len == 0 ? GetDecodeTokenNumThreshold() : input.q_seq_len - 1];
  mla_meta.num_splits = mla_shared.num_splits.GetView({batch_size + 1}, mla_shared.num_splits.shape[0]);
  mla_shared.num_splits.shape[0] += mla_meta.num_splits.GetElementNumber();
  if (!config.model_config.use_dsa) {
    llm_kernels::nvidia::FlashMlaWorkspaceMap flash_mla_workspace_map;
    flash_mla_workspace_map.num_sm_parts = mla_meta.tile_scheduler_metadata.shape[0];
    flash_mla_workspace_map.tile_scheduler_metadata_ptr = mla_meta.tile_scheduler_metadata.GetPtr<int>();
    flash_mla_workspace_map.num_splits_ptr = mla_meta.num_splits.GetPtr<int>();
    llm_kernels::nvidia::InvokeGetMlaMetadata(input.input_length.GetPtr<int>(), flash_mla_workspace_map, batch_size,
                                              config.context->device->GetH2DStream().Get());
  } else {
    mla_meta.tile_scheduler_metadata.shape[2] = 0;  // mark meta as uninitialized; DSA kernel fills it
    if (input.q_seq_len > 0) {
      dsa_meta.paged_schedule_meta = dsa_shared.paged_schedule_metas[input.q_seq_len - 1];
      dsa_meta.paged_schedule_meta.shape[2] = 0;  // mark meta as uninitialized
    }
  }
#endif
}

void PrepareChunkKV(MLASharedAttentionMeta& mla_shared, const FlashAttentionMeta& flash_meta,
                    const AttentionBuilderConfig& config) {
// TODO(zhongzhicao): kunlun平台采用两步attention，可以按照prefix的长度调度
#if defined(ENABLE_CUDA) || defined(ENABLE_KUNLUN)
  const auto& reqs = flash_meta.dp_reqs;
  const int batch_size = static_cast<int>(reqs.size());
  const int max_chunked_kv_tokens = mla_shared.max_chunked_kv_tokens;

  size_t num_chunks = 0;
  int* batch_starts_ptr = mla_shared.chunk_kv_batch_starts_host.GetPtr<int>();
  mla_shared.chunk_kv_cu_extend_lens_list.clear();
  mla_shared.chunk_kv_cu_seq_lens_list.clear();

  // Greedily pack consecutive requests into the same chunk while the summed
  // sequence length fits within max_chunked_kv_tokens; start a new chunk
  // when it would overflow.
  for (int i = 0; i < batch_size;) {
    batch_starts_ptr[num_chunks++] = i;

    int cur_extend_len_sum = 0;
    int cur_seq_len_sum = 0;
    mla_shared.chunk_kv_cu_extend_lens_list.push_back(0);
    mla_shared.chunk_kv_cu_seq_lens_list.push_back(0);

    do {
      cur_extend_len_sum += reqs[i]->GetInputIdsLength();
      cur_seq_len_sum += static_cast<int>(reqs[i]->forwarding_tokens->size());
      mla_shared.chunk_kv_cu_extend_lens_list.push_back(cur_extend_len_sum);
      mla_shared.chunk_kv_cu_seq_lens_list.push_back(cur_seq_len_sum);
      ++i;
    } while (i < batch_size &&
             cur_seq_len_sum + static_cast<int>(reqs[i]->forwarding_tokens->size()) <= max_chunked_kv_tokens);
  }
  // Sentinel: lets consumers compute the last chunk's size from consecutive differences.
  batch_starts_ptr[num_chunks++] = batch_size;
  mla_shared.chunk_kv_batch_starts_host.shape[0] = num_chunks;

  auto& stream = config.context->device->GetH2DStream();

  auto& extend_list = mla_shared.chunk_kv_cu_extend_lens_list;
  mla_shared.chunk_kv_cu_extend_lens_tensor.shape = {extend_list.size()};
  MemcpyAsync(mla_shared.chunk_kv_cu_extend_lens_tensor.GetPtr<void>(), extend_list.data(),
              extend_list.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE, stream);

  auto& seq_list = mla_shared.chunk_kv_cu_seq_lens_list;
  mla_shared.chunk_kv_cu_seq_lens_tensor.shape = {seq_list.size()};
  MemcpyAsync(mla_shared.chunk_kv_cu_seq_lens_tensor.GetPtr<void>(), seq_list.data(), seq_list.size() * sizeof(int),
              MEMCPY_HOST_TO_DEVICE, stream);
#endif
}

}  // namespace ksana_llm
