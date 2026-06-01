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
  if (GetDecodeTokenNumThreshold() > 1) {
    expanded_c4_seq_lens_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size, GetDecodeTokenNumThreshold()});
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
  flash_meta.Reset();
  page_metas.clear();
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
        flash_meta.dp_reqs.emplace_back(&req);
        flash_meta.total_dp_input_ids_len += extend_len;
        flash_meta.total_dp_compressed_input_ids_len += compressed_extend_len;
        dp_context_tokens += extend_len;
        dp_context_kv_tokens += compressed_extend_len;
      } else {  // GetForwardRequestType(req) == ForwardRequestType::kPage
        if (page_metas.empty() || page_metas.back().q_seq_len != extend_len) {
          page_metas.emplace_back().q_seq_len = extend_len;
        }
        auto& page_meta = page_metas.back();
        page_meta.dp_reqs.emplace_back(&req);
        page_meta.total_dp_input_ids_len += extend_len;
        page_meta.total_dp_compressed_input_ids_len += compressed_extend_len;
        dp_decode_tokens += extend_len;
      }
    }
  }

  if (!flash_meta.dp_reqs.empty()) {
    // Delegate flash chunking to internal IndexerMeta (compress_ratio-aware)
    indexer_meta->PrepareChunk(flash_meta);
    indexer_meta->PrepareCuSeqLen(flash_meta);
    indexer_meta->PrepareBatchIndices(flash_meta);

    for (const auto* req : flash_meta.dp_reqs) {
      PrepareExtendLens(flash_meta.extend_lens_host, *req);
    }
    HOST_TO_SHARED_DEVICE(extend_lens, flash_meta);
  }

  for (auto& page_meta : page_metas) {
    // Delegate paged cu_seq_len to IndexerMeta
    indexer_meta->PrepareCuSeqLen(page_meta, /*is_paged=*/true);

    PrepareC4SeqLens(page_meta);
  }
}

void C4IndexerMeta::PrepareC4SeqLens(input_meta& input) {
  // Build c4_seq_lens and expanded_c4_seq_lens for all requests in this page_input.
  // Shape: c4_seq_lens [reqs.size()], expanded_c4_seq_lens [reqs.size() * q_seq_len].
  const auto& reqs = input.dp_reqs;
  const size_t q_seq_len = input.q_seq_len;

  for (const auto* req : reqs) {
    input.c4_seq_lens_host.push_back(static_cast<int>(req->forwarding_tokens->size() / kC4Ratio));
    if (q_seq_len > 1) {
      for (size_t i = 1; i <= q_seq_len; ++i) {
        input.expanded_c4_seq_lens_host.push_back((req->kv_cached_token_num + i) / kC4Ratio);
      }
    }
  }

  HOST_TO_SHARED_DEVICE(c4_seq_lens, input);
  if (q_seq_len > 1) {
    HOST_TO_SHARED_DEVICE(expanded_c4_seq_lens, input);
  }
}

void C4IndexerMeta::PrepareExtendLens(std::vector<int>& host, const ForwardRequest& req) {
  host.push_back(static_cast<int>(req.GetInputIdsLength()));
}

}  // namespace ksana_llm
