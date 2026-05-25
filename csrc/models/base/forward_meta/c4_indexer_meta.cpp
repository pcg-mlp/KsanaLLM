/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/base/forward_meta/c4_indexer_meta.h"

#include "device/device_utils.h"
#include "models/base/forward_meta/common.h"
#include "utils/logger.h"

namespace ksana_llm {

C4IndexerMeta::C4IndexerMeta(const RuntimeConfig& runtime_config, int rank, std::shared_ptr<Context> context)
    : context_(context) {
  attn_dp_group_id_ = rank / runtime_config.parallel_basic_config.attn_tensor_parallel_size;

  // Dummy tensor for indexer interface (store_out_loc is not used for compress_ratio==4)
  gathered_c4_out_loc_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {1});
  const size_t max_batch_size = runtime_config.max_batch_size;
  c4_seq_lens_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  extend_lens_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size});
  // DeepGEMM fp8_paged_mqa_logits kernel only supports q_seq_len in {1, 2}.
  // For MTP > 2, we "fold" the query sequence dimension into the batch dimension:
  // - Original: batch_size=B, q_seq_len=N (where N > 2)
  // - Expanded: batch_size=B*N, q_seq_len=1 (if N is odd)
  //          OR batch_size=B*N/2, q_seq_len=2 (if N is even)
  // This requires expanding input_length tensors accordingly.
  constexpr size_t kDeepGemmMaxQSeqLen = 2;
  if (GetDecodeTokenNumThreshold() > kDeepGemmMaxQSeqLen) {
    expanded_c4_seq_lens_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size * GetDecodeTokenNumThreshold()});
  }

  indexer_meta = std::make_unique<IndexerMeta>(runtime_config, rank, context, kC4Ratio);
}

void C4IndexerMeta::Prepare(const std::vector<ForwardRequest>& forward_reqs) {
  // Reset the offsets of shared tensors
  c4_seq_lens_tensor.shape = {0};
  if (!expanded_c4_seq_lens_tensor.shape.empty()) {
    expanded_c4_seq_lens_tensor.shape = {0};
  }
  extend_lens_tensor.shape = {0};
  indexer_meta->cu_seq_len_start_tensor.shape = {0};
  indexer_meta->cu_seq_len_end_tensor.shape = {0};
  indexer_meta->batch_indices_tensor.shape = {0};

  // Reset meta
  flash_input.Reset();
  page_inputs.clear();
  dp_context_tokens = 0;
  dp_decode_tokens = 0;
  dp_context_kv_tokens = 0;

  // Categorize requests
  for (const auto& req : forward_reqs) {
    if (req.attn_dp_group_id == attn_dp_group_id_) {
      const size_t seq_len = req.forwarding_tokens->size();
      const size_t extend_len = req.GetInputIdsLength();
      const size_t prefix_len = seq_len - extend_len;
      const size_t compressed_extend_len = seq_len / kC4Ratio - prefix_len / kC4Ratio;
      if (GetForwardRequestType(req) == ForwardRequestType::kFlash) {
        flash_input.dp_reqs.emplace_back(&req);
        flash_input.total_dp_input_ids_len += extend_len;
        flash_input.total_dp_compressed_input_ids_len += compressed_extend_len;
        dp_context_tokens += extend_len;
        dp_context_kv_tokens += compressed_extend_len;
      } else {  // GetForwardRequestType(req) == ForwardRequestType::kPage
        if (page_inputs.empty() || page_inputs.back().q_seq_len != extend_len) {
          // `page_inputs.back().q_seq_len` should be greater than `extend_len`,
          // since requests in the same dp group are sorted in descending order by their token numbers
          page_inputs.emplace_back().q_seq_len = extend_len;
          // For odd q_seq_len, fold into q_seq_len batches with q_seq_len=1
          // For even q_seq_len, fold into q_seqlen/2 batches with q_seq_len=2
          auto& page_input = page_inputs.back();
          if (extend_len & 1) {
            page_input.expanded_q_seq_len = 1;
            page_input.expanded_batch_size = extend_len;
          } else {
            page_input.expanded_q_seq_len = 2;
            page_input.expanded_batch_size = extend_len >> 1;
          }
        }
        auto& page_input = page_inputs.back();
        page_input.dp_reqs.emplace_back(&req);
        page_input.total_dp_input_ids_len += extend_len;
        page_input.total_dp_compressed_input_ids_len += compressed_extend_len;
        dp_decode_tokens += extend_len;
      }
    }
  }

  if (!flash_input.dp_reqs.empty()) {
    // Delegate flash chunking to internal IndexerMeta (compress_ratio-aware)
    indexer_meta->PrepareChunk(flash_input);
    indexer_meta->PrepareCuSeqLen(flash_input);
    indexer_meta->PrepareBatchIndices(flash_input);

    for (const auto* req : flash_input.dp_reqs) {
      PrepareExtendLens(flash_input.extend_lens_host, *req);
    }
    HOST_TO_SHARED_DEVICE(extend_lens, flash_input);
  }

  for (auto& page_input : page_inputs) {
    // Delegate paged cu_seq_len to IndexerMeta
    indexer_meta->PrepareCuSeqLen(page_input, /*is_paged=*/true);

    for (const auto* req : page_input.dp_reqs) {
      PrepareC4SeqLens(page_input.c4_seq_lens_host, page_input.expanded_c4_seq_lens_host, *req,
                       page_input.expanded_q_seq_len, page_input.expanded_batch_size);
    }
    HOST_TO_SHARED_DEVICE(c4_seq_lens, page_input);
    if (!expanded_c4_seq_lens_tensor.shape.empty()) {
      HOST_TO_SHARED_DEVICE(expanded_c4_seq_lens, page_input);
    }
  }
}

void C4IndexerMeta::PrepareC4SeqLens(std::vector<int>& host, std::vector<int>& expanded_host, const ForwardRequest& req,
                                     size_t expanded_q_seq_len, size_t expanded_batch_size) {
  host.push_back(static_cast<int>(req.forwarding_tokens->size() / kC4Ratio));

  // Prepare expanded version if needed
  if (!expanded_c4_seq_lens_tensor.shape.empty()) {
    for (size_t i = 1; i <= expanded_batch_size; ++i) {
      expanded_host.push_back((req.kv_cached_token_num + i * expanded_q_seq_len) / kC4Ratio);
    }
  }
}

void C4IndexerMeta::PrepareExtendLens(std::vector<int>& host, const ForwardRequest& req) {
  host.push_back(static_cast<int>(req.GetInputIdsLength()));
}

}  // namespace ksana_llm
