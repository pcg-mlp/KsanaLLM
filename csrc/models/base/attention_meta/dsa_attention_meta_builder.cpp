/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/attention_meta/dsa_attention_meta_builder.h"

#include <algorithm>

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/deepseek_deepgemm_wrapper.h"
#endif
#include "profiler/profile_event.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
using ksana_llm::nvidia::kDeepGemmMaxQSeqLen;
#else
constexpr int kDeepGemmMaxQSeqLen = 2;
#endif

void PrepareStoreOutLoc(DSAAttentionMeta& dsa_meta, DSASharedAttentionMeta& dsa_shared, const AttentionMeta& input,
                        const AttentionBuilderConfig& config) {
  PROFILE_EVENT_SCOPE(PrepareStoreOutLoc, "PrepareStoreOutLoc", config.rank);
  const int block_token_num = config.runtime_config.attn_backend_config.block_token_num;
  std::vector<int> store_out_loc_host;
  store_out_loc_host.reserve(input.total_dp_input_ids_len);
  for (const auto* req : input.dp_reqs) {
    const int seq_len = req->forwarding_tokens->size();
    const int extend_len = req->GetInputIdsLength();
    const int prefix_len = seq_len - extend_len;
    const auto& block_ids = req->atb_kv_cache_base_blk_ids[config.attn_dp_rank_id];
    for (int i = prefix_len; i < seq_len; ++i) {
      store_out_loc_host.push_back(block_ids[i / block_token_num] * block_token_num + i % block_token_num);
    }
  }
  dsa_meta.store_out_loc =
      dsa_shared.store_out_loc.GetView({store_out_loc_host.size()}, dsa_shared.store_out_loc.shape[0]);
  dsa_shared.store_out_loc.shape[0] += dsa_meta.store_out_loc.GetElementNumber();
  MemcpyAsync(dsa_meta.store_out_loc.GetPtr<void>(), store_out_loc_host.data(), store_out_loc_host.size() * sizeof(int),
              MEMCPY_HOST_TO_DEVICE, config.context->device->GetH2DStream());
}

void PrepareDSAExpandedBlockTable(DSAAttentionMeta& dsa_meta, DSASharedAttentionMeta& dsa_shared,
                                  const AttentionMeta& input, const AttentionBuilderConfig& config) {
  if (!config.enable_blocked_multi_token_forwarding_kv && !config.use_flashinfer_for_decode) {
    return;
  }
  if (input.q_seq_len <= static_cast<size_t>(kDeepGemmMaxQSeqLen)) {
    return;
  }
  PROFILE_EVENT_SCOPE(PrepareDSAExpandedBlockTable, "PrepareDSAExpandedBlockTable", config.rank);
  const auto& reqs = input.dp_reqs;

  size_t max_num_blocks_per_query = 0;
  for (const auto& req : reqs) {
    max_num_blocks_per_query =
        std::max(max_num_blocks_per_query, req->atb_kv_cache_base_blk_ids[config.attn_dp_rank_id].size());
  }

  std::vector<int32_t> expanded_block_table_host(reqs.size() * input.expanded_batch_size * max_num_blocks_per_query,
                                                 -1);
  for (size_t i = 0; i < reqs.size(); ++i) {
    const auto& block_ids = reqs[i]->atb_kv_cache_base_blk_ids[config.attn_dp_rank_id];
    for (size_t j = 0; j < input.expanded_batch_size; ++j) {
      for (size_t base_block_idx = 0; base_block_idx < block_ids.size(); ++base_block_idx) {
        expanded_block_table_host[(i * input.expanded_batch_size + j) * max_num_blocks_per_query + base_block_idx] =
            block_ids[base_block_idx];
      }
    }
  }
  KLLM_LOG_DEBUG << "expanded_block_table_host " << expanded_block_table_host;
  dsa_meta.expanded_block_table = dsa_shared.expanded_block_table.GetView(
      {reqs.size() * input.expanded_batch_size, max_num_blocks_per_query}, dsa_shared.expanded_block_table.shape[0]);
  dsa_shared.expanded_block_table.shape[0] += dsa_meta.expanded_block_table.GetElementNumber();
  MemcpyAsync(dsa_meta.expanded_block_table.GetPtr<void>(), expanded_block_table_host.data(),
              expanded_block_table_host.size() * sizeof(decltype(expanded_block_table_host)::value_type),
              MEMCPY_HOST_TO_DEVICE, config.context->device->GetH2DStream());
}

}  // namespace ksana_llm
