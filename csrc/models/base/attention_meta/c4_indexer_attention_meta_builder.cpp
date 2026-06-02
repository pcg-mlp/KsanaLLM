/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/base/attention_meta/c4_indexer_attention_meta_builder.h"

#include "utils/logger.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

static constexpr size_t kC4Ratio = 4;

// Upload a host vector to a shared device tensor, creating a view and issuing H2D memcpy.
template <typename T>
static void HostToSharedDeviceC4(const std::vector<T>& host_vec, Tensor& shared_tensor, Tensor& meta_tensor,
                                  const AttentionBuilderConfig& config) {
  meta_tensor = shared_tensor.GetView({host_vec.size()}, shared_tensor.shape[0]);
  shared_tensor.shape[0] += meta_tensor.GetElementNumber();
  MemcpyAsync(meta_tensor.GetPtr<void>(), host_vec.data(), host_vec.size() * sizeof(T), MEMCPY_HOST_TO_DEVICE,
              config.context->device->GetH2DStream());
}

// Build extend_lens (number of new tokens per flash request) and upload to device.
// Mirrors C4IndexerMeta::PrepareExtendLens.
static void PrepareFlashExtendLens(C4IndexerAttentionMeta& indexer_meta, C4IndexerSharedAttentionMeta& indexer_shared,
                                    const std::vector<ForwardRequest*>& dp_reqs,
                                    const AttentionBuilderConfig& config) {
  indexer_meta.extend_lens_host.clear();
  indexer_meta.extend_lens_host.reserve(dp_reqs.size());
  for (const auto* req : dp_reqs) {
    indexer_meta.extend_lens_host.push_back(static_cast<int>(req->GetInputIdsLength()));
  }
  KLLM_LOG_DEBUG << "extend_lens_host: " << indexer_meta.extend_lens_host;
  HostToSharedDeviceC4(indexer_meta.extend_lens_host, indexer_shared.extend_lens_tensor,
                       indexer_meta.extend_lens_tensor, config);
}

// Build c4_seq_lens (KV length in compressed space) and optionally expanded_c4_seq_lens.
// Mirrors C4IndexerMeta::PrepareC4SeqLens.
static void PrepareC4SeqLens(C4IndexerAttentionMeta& indexer_meta, C4IndexerSharedAttentionMeta& indexer_shared,
                              const std::vector<ForwardRequest*>& dp_reqs, size_t q_seq_len,
                              const AttentionBuilderConfig& config) {
  for (const auto* req : dp_reqs) {
    indexer_meta.c4_seq_lens_host.push_back(static_cast<int>(req->forwarding_tokens->size() / kC4Ratio));
    if (q_seq_len > 1) {
      for (size_t i = 1; i <= q_seq_len; ++i) {
        indexer_meta.expanded_c4_seq_lens_host.push_back(
            static_cast<int>((req->kv_cached_token_num + i) / kC4Ratio));
      }
    }
  }
  KLLM_LOG_DEBUG << "c4_seq_lens_host: " << indexer_meta.c4_seq_lens_host;
  HostToSharedDeviceC4(indexer_meta.c4_seq_lens_host, indexer_shared.c4_seq_lens_tensor,
                       indexer_meta.c4_seq_lens_tensor, config);
  if (q_seq_len > 1) {
    KLLM_LOG_DEBUG << "expanded_c4_seq_lens_host: " << indexer_meta.expanded_c4_seq_lens_host;
    HostToSharedDeviceC4(indexer_meta.expanded_c4_seq_lens_host, indexer_shared.expanded_c4_seq_lens_tensor,
                         indexer_meta.expanded_c4_seq_lens_tensor, config);
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

void PrepareFlashC4Indexer(C4IndexerAttentionMeta& indexer_meta, C4IndexerSharedAttentionMeta& indexer_shared,
                            const FlashAttentionMeta& flash_meta, const AttentionBuilderConfig& config) {
  // Base path: prepares cu_seq_len_{start,end}, batch_indices, chunk data.
  PrepareFlashIndexer(indexer_meta, indexer_shared, flash_meta, config);

  // C4-specific: compute compressed extend count and dp_context_kv_tokens.
  indexer_meta.total_dp_compressed_input_ids_len = 0;
  for (const auto* req : flash_meta.dp_reqs) {
    const size_t seq_len = req->forwarding_tokens->size();
    const size_t extend_len = req->GetInputIdsLength();
    const size_t prefix_len = seq_len - extend_len;
    const size_t compressed_extend_len = seq_len / kC4Ratio - prefix_len / kC4Ratio;
    indexer_meta.total_dp_compressed_input_ids_len += compressed_extend_len;
    indexer_shared.dp_context_kv_tokens += compressed_extend_len;
  }

  PrepareFlashExtendLens(indexer_meta, indexer_shared, flash_meta.dp_reqs, config);
}

void PreparePagedC4Indexer(C4IndexerAttentionMeta& indexer_meta, C4IndexerSharedAttentionMeta& indexer_shared,
                            const PagedAttentionMeta& page_meta, const AttentionBuilderConfig& config) {
  // Base path: prepares cu_seq_len_{start,end} (paged) and expanded_seq_len.
  PreparePagedIndexer(indexer_meta, indexer_shared, page_meta, config);

  // C4-specific: compressed extend count.
  indexer_meta.total_dp_compressed_input_ids_len = 0;
  for (const auto* req : page_meta.dp_reqs) {
    const size_t seq_len = req->forwarding_tokens->size();
    const size_t extend_len = req->GetInputIdsLength();
    const size_t prefix_len = seq_len - extend_len;
    indexer_meta.total_dp_compressed_input_ids_len += seq_len / kC4Ratio - prefix_len / kC4Ratio;
  }

  PrepareC4SeqLens(indexer_meta, indexer_shared, page_meta.dp_reqs, page_meta.q_seq_len, config);
}

}  // namespace ksana_llm
