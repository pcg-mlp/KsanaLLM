/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/attention_meta/indexer_attention_meta_builder.h"

#include <fmt/core.h>

#include <algorithm>

#include "utils/logger.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

// ── Internal helpers ──────────────────────────────────────────────────────────
// These mirror IndexerMeta::PrepareChunk/PrepareCuSeqLen/PrepareBatchIndices/PrepareSeqLen
// but work on free-function arguments instead of class members.

// Macro-less equivalent of HOST_TO_SHARED_DEVICE: creates a view from `shared_tensor`,
// advances its shape[0], and issues an async H2D memcpy of `host_vec` to the view.
template <typename T>
static void HostToSharedDevice(const std::vector<T>& host_vec, Tensor& shared_tensor, Tensor& meta_tensor,
                                const AttentionBuilderConfig& config) {
  meta_tensor = shared_tensor.GetView({host_vec.size()}, shared_tensor.shape[0]);
  shared_tensor.shape[0] += meta_tensor.GetElementNumber();
  MemcpyAsync(meta_tensor.GetPtr<void>(), host_vec.data(), host_vec.size() * sizeof(T), MEMCPY_HOST_TO_DEVICE,
              config.context->device->GetH2DStream());
}

// Builds chunk Q-boundaries and KV ranges into indexer_shared host tensors.
// Mirrors IndexerMeta::PrepareChunk().
static void PrepareChunk(IndexerSharedAttentionMeta& indexer_shared, IndexerAttentionMeta& /*indexer_meta*/,
                         const FlashAttentionMeta& flash_meta) {
  const auto& reqs = flash_meta.dp_reqs;
  const int batch_size = static_cast<int>(reqs.size());
  const int compress_ratio = indexer_shared.compress_ratio;
  const int max_chunked_q_tokens = indexer_shared.max_chunked_q_tokens;
  const int max_chunked_kv_tokens = indexer_shared.max_chunked_kv_tokens;

  int num_chunks = 0;
  int* token_starts_ptr = indexer_shared.chunk_token_starts_host.GetPtr<int>();
  int* kv_token_ranges_ptr = indexer_shared.chunk_kv_token_ranges_host.GetPtr<int>();
  token_starts_ptr[0] = 0;
  num_chunks++;

  int cu_extend_len = 0;
  int cu_compressed_seq_len = 0;
  int cur_q_tokens = 0;
  int cur_kv_tokens = 0;

  auto CompressedKV = [&](int pos) { return pos / compress_ratio; };

  auto CloseLastChunk = [&](int q_end, int kv_end) {
    token_starts_ptr[num_chunks] = q_end;
    const int chunk_idx = num_chunks - 1;
    kv_token_ranges_ptr[2 * chunk_idx] = kv_end - cur_kv_tokens;
    kv_token_ranges_ptr[2 * chunk_idx + 1] = kv_end;
    KLLM_LOG_DEBUG << fmt::format("Indexer chunk({}): Q [{}, {}), KV [{}, {})", chunk_idx,
                                  token_starts_ptr[num_chunks - 1], token_starts_ptr[num_chunks],
                                  kv_token_ranges_ptr[2 * chunk_idx], kv_token_ranges_ptr[2 * chunk_idx + 1]);
    num_chunks++;
    cur_q_tokens = 0;
    cur_kv_tokens = 0;
  };

  for (int r = 0; r < batch_size; ++r) {
    const int extend_len = static_cast<int>(reqs[r]->GetInputIdsLength());
    const int seq_len = static_cast<int>(reqs[r]->forwarding_tokens->size());
    const int prefix_len = seq_len - extend_len;
    const int compressed_prefix_len = CompressedKV(prefix_len);

    if (cur_kv_tokens + compressed_prefix_len >= max_chunked_kv_tokens) {
      CloseLastChunk(cu_extend_len, cu_compressed_seq_len);
    }

    cur_kv_tokens += compressed_prefix_len;

    int taken = 0;
    int compressed_taken = compressed_prefix_len;
    while (taken < extend_len) {
      const int remaining = extend_len - taken;
      const int remaining_kv = max_chunked_kv_tokens - cur_kv_tokens;
      const int can_take_by_kv =
          ((compressed_taken + remaining_kv + 1) * compress_ratio - 1) - (prefix_len + taken);
      const int can_take = std::min({remaining, max_chunked_q_tokens - cur_q_tokens, can_take_by_kv});

      taken += can_take;
      cur_q_tokens += can_take;
      const int prev = compressed_taken;
      compressed_taken = CompressedKV(prefix_len + taken);
      cur_kv_tokens += compressed_taken - prev;

      if (can_take < remaining) {
        CloseLastChunk(cu_extend_len + taken, cu_compressed_seq_len + compressed_taken);
        cur_kv_tokens = compressed_taken;
      }
    }

    cu_extend_len += extend_len;
    cu_compressed_seq_len += CompressedKV(seq_len);
  }

  CloseLastChunk(cu_extend_len, cu_compressed_seq_len);

  indexer_shared.chunk_token_starts_host.shape[0] = num_chunks;
  indexer_shared.chunk_kv_token_ranges_host.shape[0] = 2 * (num_chunks - 1);
}

// Builds per-token KV range [start, end) host arrays and uploads to device.
// Mirrors IndexerMeta::PrepareCuSeqLen().
static void PrepareCuSeqLen(IndexerAttentionMeta& indexer_meta, IndexerSharedAttentionMeta& indexer_shared,
                             const std::vector<ForwardRequest*>& dp_reqs, bool is_paged,
                             const AttentionBuilderConfig& config) {
  const int compress_ratio = indexer_shared.compress_ratio;
  indexer_meta.cu_seq_len_start_host.clear();
  indexer_meta.cu_seq_len_end_host.clear();

  if (!is_paged) {
    const int* token_starts_ptr = indexer_shared.chunk_token_starts_host.GetPtr<int>();
    const size_t num_chunks = indexer_shared.chunk_token_starts_host.shape[0] - 1;
    const int batch_size = static_cast<int>(dp_reqs.size());

    int r = 0;
    int cu_extend_len = 0;
    size_t c = 0;

    while (r < batch_size && c < num_chunks) {
      const int chunk_start = token_starts_ptr[c];
      const int chunk_end = token_starts_ptr[c + 1];
      const int extend_len = static_cast<int>(dp_reqs[r]->GetInputIdsLength());
      const int seq_len = static_cast<int>(dp_reqs[r]->forwarding_tokens->size());
      const int prefix_len = seq_len - extend_len;
      const int req_end = cu_extend_len + extend_len;

      const int overlap_start = std::max(cu_extend_len, chunk_start);
      const int overlap_end = std::min(req_end, chunk_end);

      if (overlap_start < overlap_end) {
        for (int i = overlap_start - cu_extend_len + 1; i <= overlap_end - cu_extend_len; ++i) {
          indexer_meta.cu_seq_len_start_host.push_back(0);
          indexer_meta.cu_seq_len_end_host.push_back((prefix_len + i) / compress_ratio);
        }
      }

      if (req_end <= chunk_end) {
        cu_extend_len += extend_len;
        r++;
      } else {
        c++;
      }
    }
  } else {
    for (const auto* req : dp_reqs) {
      const int seq_len = static_cast<int>(req->forwarding_tokens->size());
      const int extend_len = static_cast<int>(req->GetInputIdsLength());
      const int prefix_len = seq_len - extend_len;
      for (int i = 1; i <= extend_len; ++i) {
        indexer_meta.cu_seq_len_start_host.push_back(0);
        indexer_meta.cu_seq_len_end_host.push_back((prefix_len + i) / compress_ratio);
      }
    }
  }

  KLLM_LOG_DEBUG << "cu_seq_len_start_host: " << indexer_meta.cu_seq_len_start_host;
  HostToSharedDevice(indexer_meta.cu_seq_len_start_host, indexer_shared.cu_seq_len_start_tensor,
                     indexer_meta.cu_seq_len_start_tensor, config);
  KLLM_LOG_DEBUG << "cu_seq_len_end_host: " << indexer_meta.cu_seq_len_end_host;
  HostToSharedDevice(indexer_meta.cu_seq_len_end_host, indexer_shared.cu_seq_len_end_tensor,
                     indexer_meta.cu_seq_len_end_tensor, config);
}

// Builds per-token batch index array.
// Mirrors IndexerMeta::PrepareBatchIndices().
static void PrepareBatchIndices(IndexerAttentionMeta& indexer_meta, IndexerSharedAttentionMeta& indexer_shared,
                                 const std::vector<ForwardRequest*>& dp_reqs, const AttentionBuilderConfig& config) {
  indexer_meta.batch_indices_host.clear();
  indexer_meta.batch_indices_host.reserve(indexer_meta.total_dp_input_ids_len);
  const int batch_size = static_cast<int>(dp_reqs.size());
  for (int r = 0; r < batch_size; ++r) {
    const int extend_len = static_cast<int>(dp_reqs[r]->GetInputIdsLength());
    for (int i = 0; i < extend_len; ++i) {
      indexer_meta.batch_indices_host.push_back(r);
    }
  }
  KLLM_LOG_DEBUG << "batch_indices_host: " << indexer_meta.batch_indices_host;
  HostToSharedDevice(indexer_meta.batch_indices_host, indexer_shared.batch_indices_tensor,
                     indexer_meta.batch_indices_tensor, config);
}

// Builds expanded context length for DeepGEMM paged path (q_seq_len > 1).
// Mirrors IndexerMeta::PrepareSeqLen().
static void PrepareSeqLen(IndexerAttentionMeta& indexer_meta, IndexerSharedAttentionMeta& indexer_shared,
                           const std::vector<ForwardRequest*>& dp_reqs, size_t q_seq_len,
                           const AttentionBuilderConfig& config) {
  indexer_meta.expanded_seq_len_host.clear();
  indexer_meta.expanded_seq_len_host.reserve(dp_reqs.size() * q_seq_len);
  for (const auto* req : dp_reqs) {
    for (size_t j = 1; j <= q_seq_len; ++j) {
      indexer_meta.expanded_seq_len_host.push_back(static_cast<int>(req->kv_cached_token_num + j));
    }
  }
  KLLM_LOG_DEBUG << "expanded_seq_len_host: " << indexer_meta.expanded_seq_len_host;
  HostToSharedDevice(indexer_meta.expanded_seq_len_host, indexer_shared.expanded_seq_len_tensor,
                     indexer_meta.expanded_seq_len_tensor, config);
}

// ── Public API ────────────────────────────────────────────────────────────────

void PrepareFlashIndexer(IndexerAttentionMeta& indexer_meta, IndexerSharedAttentionMeta& indexer_shared,
                         const FlashAttentionMeta& flash_meta, const AttentionBuilderConfig& config) {
  indexer_meta.Reset();
  indexer_meta.total_dp_input_ids_len = flash_meta.total_dp_input_ids_len;
  indexer_shared.dp_context_tokens += flash_meta.total_dp_input_ids_len;

  PrepareChunk(indexer_shared, indexer_meta, flash_meta);
  PrepareCuSeqLen(indexer_meta, indexer_shared, flash_meta.dp_reqs, /*is_paged=*/false, config);
  PrepareBatchIndices(indexer_meta, indexer_shared, flash_meta.dp_reqs, config);
}

void PreparePagedIndexer(IndexerAttentionMeta& indexer_meta, IndexerSharedAttentionMeta& indexer_shared,
                         const PagedAttentionMeta& page_meta, const AttentionBuilderConfig& config) {
  indexer_meta.Reset();
  indexer_meta.total_dp_input_ids_len = page_meta.total_dp_input_ids_len;
  indexer_shared.dp_decode_tokens += page_meta.total_dp_input_ids_len;

  if (page_meta.q_seq_len > 1) {
    PrepareSeqLen(indexer_meta, indexer_shared, page_meta.dp_reqs, page_meta.q_seq_len, config);
  }
  PrepareCuSeqLen(indexer_meta, indexer_shared, page_meta.dp_reqs, /*is_paged=*/true, config);
}

}  // namespace ksana_llm
