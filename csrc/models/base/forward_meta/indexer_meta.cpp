/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/base/forward_meta/indexer_meta.h"

#include <fmt/core.h>

#include "device/device_utils.h"
#include "models/base/forward_meta/common.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

IndexerMeta::IndexerMeta(const RuntimeConfig& runtime_config, int rank, std::shared_ptr<Context> context,
                         int compress_ratio)
    : context_(context), compress_ratio_(compress_ratio) {
  KLLM_CHECK_WITH_INFO(compress_ratio_ == 1 || compress_ratio_ == 4, "IndexerMeta requires compress_ratio in {1, 4}");
  attn_dp_group_id_ = rank / runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  max_chunked_q_tokens_ =
      ComputeIndexerMaxQChunkSize(runtime_config.max_seq_len / compress_ratio_, runtime_config.max_step_token_num);
  max_chunked_kv_tokens_ = runtime_config.max_chunked_kv_tokens;
  KLLM_LOG_INFO << fmt::format("IndexerMeta: max_chunked_q_tokens={}, max_chunked_kv_tokens={}, compress_ratio={}",
                               max_chunked_q_tokens_, max_chunked_kv_tokens_, compress_ratio_);

  const size_t max_batch_size = runtime_config.max_batch_size;
  chunk_token_starts_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {max_batch_size + 1});
  chunk_kv_token_ranges_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {max_batch_size * 2});
  const size_t max_token_num = runtime_config.max_step_token_num;
  cu_seq_len_start_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  cu_seq_len_end_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  batch_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num});
  if (GetDecodeTokenNumThreshold() > 1) {
    expanded_seq_len_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size, GetDecodeTokenNumThreshold()});
  }
}

void IndexerMeta::Prepare(const std::vector<ForwardRequest>& forward_reqs) {
  // Reset shared tensor offsets
  cu_seq_len_start_tensor.shape = {0};
  cu_seq_len_end_tensor.shape = {0};
  batch_indices_tensor.shape = {0};
  if (!expanded_seq_len_tensor.shape.empty()) {
    expanded_seq_len_tensor.shape = {0};
  }

  // Reset meta
  flash_meta.Reset();
  page_metas.clear();
  dp_context_tokens = 0;
  dp_decode_tokens = 0;

  // Categorize requests into flash (prefill) vs page (decode)
  for (const auto& req : forward_reqs) {
    if (req.attn_dp_group_id == attn_dp_group_id_) {
      const size_t extend_len = req.GetInputIdsLength();
      if (GetForwardRequestType(req) == ForwardRequestType::kFlash) {
        flash_meta.dp_reqs.emplace_back(&req);
        flash_meta.total_dp_input_ids_len += extend_len;
        dp_context_tokens += extend_len;
      } else {
        if (page_metas.empty() || page_metas.back().q_seq_len != extend_len) {
          page_metas.emplace_back().q_seq_len = extend_len;
        }
        auto& page_meta = page_metas.back();
        page_meta.dp_reqs.emplace_back(&req);
        page_meta.total_dp_input_ids_len += extend_len;
        dp_decode_tokens += extend_len;
      }
    }
  }

  if (!flash_meta.dp_reqs.empty()) {
    PrepareChunk(flash_meta);
    PrepareCuSeqLen(flash_meta);
    PrepareBatchIndices(flash_meta);
  }
  for (auto& page_meta : page_metas) {
    if (page_meta.q_seq_len > 1) {
      PrepareSeqLen(page_meta);
    }
    PrepareCuSeqLen(page_meta, /*is_paged=*/true);
  }
}

void IndexerMeta::PrepareChunk(input_meta& input) {
  const auto& reqs = input.dp_reqs;
  const int batch_size = reqs.size();

  int num_chunks = 0;
  // Q token boundaries (cumstart array). Adjacent chunks have contiguous Q tokens (no gap/overlap).
  int* token_starts_ptr = chunk_token_starts_host.GetPtr<int>();
  // KV token [start, end) ranges per chunk — in COMPRESSED space.
  // Adjacent chunks may have overlapping KV tokens (due to shared prefix).
  int* kv_token_ranges_ptr = chunk_kv_token_ranges_host.GetPtr<int>();
  token_starts_ptr[0] = 0;
  num_chunks++;

  int cu_extend_len = 0;          // cumulative extend_len (Q space, each request counted once)
  int cu_compressed_seq_len = 0;  // cumulative compressed KV entries (each request counted once)
  int cur_q_tokens = 0;           // total Q tokens of current chunk
  int cur_kv_tokens = 0;          // total KV tokens of current chunk (compressed space)

  // Helper to compute number of compressed KV entries spanning [0, pos) (0-indexed, pos = prefix+extend consumed)
  auto CompressedKV = [&](int pos) { return pos / compress_ratio_; };

  // Close the current (last open) chunk: record its Q boundary and KV range, then reset counters.
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
    // Compressed KV entries for prefix_len positions.
    const int compressed_prefix_len = CompressedKV(prefix_len);

    // Close current chunk if adding this request's prefix KV would exceed budget.
    if (cur_kv_tokens + compressed_prefix_len >= max_chunked_kv_tokens_) {
      CloseLastChunk(cu_extend_len, cu_compressed_seq_len);
    }

    cur_kv_tokens += compressed_prefix_len;

    int taken = 0;
    int compressed_taken = compressed_prefix_len;
    while (taken < extend_len) {
      const int remaining = extend_len - taken;

      const int remaining_kv = max_chunked_kv_tokens_ - cur_kv_tokens;
      // Max Q tokens allowed by remaining KV budget.
      // Derived from: CompressedKV(prefix_len + taken + can_take) ≤ compressed_taken + remaining_kv.
      const int can_take_by_kv = ((compressed_taken + remaining_kv + 1) * compress_ratio_ - 1) - (prefix_len + taken);

      const int can_take = std::min({remaining, max_chunked_q_tokens_ - cur_q_tokens, can_take_by_kv});

      taken += can_take;
      cur_q_tokens += can_take;
      const int prev = compressed_taken;
      compressed_taken = CompressedKV(prefix_len + taken);
      cur_kv_tokens += compressed_taken - prev;

      if (can_take < remaining) {
        CloseLastChunk(cu_extend_len + taken, cu_compressed_seq_len + compressed_taken);
        // The next chunk still requires the full KV.
        cur_kv_tokens = compressed_taken;
      }
    }

    cu_extend_len += extend_len;
    cu_compressed_seq_len += CompressedKV(seq_len);
  }

  // Sentinel: close the last chunk.
  CloseLastChunk(cu_extend_len, cu_compressed_seq_len);

  chunk_token_starts_host.shape[0] = num_chunks;
  chunk_kv_token_ranges_host.shape[0] = 2 * (num_chunks - 1);
}

void IndexerMeta::PrepareCuSeqLen(input_meta& input, bool is_paged) {
  input.cu_seq_len_start_host.clear();
  input.cu_seq_len_end_host.clear();

  if (!is_paged) {
    // Flash path: linear two-pointer merge over chunks and requests.
    // Both chunk Q-ranges and request Q-ranges are sorted and contiguous, so we merge
    // them in a single pass — advancing the request pointer when a request is fully
    // consumed, or advancing the chunk pointer when a chunk is fully consumed.
    // PrepareChunk() must be called first.
    const int* token_starts_ptr = chunk_token_starts_host.GetPtr<int>();
    const size_t num_chunks = chunk_token_starts_host.shape[0] - 1;
    const auto& reqs = input.dp_reqs;
    const int batch_size = static_cast<int>(reqs.size());

    int r = 0;              // request pointer (advances monotonically)
    int cu_extend_len = 0;  // cumulative extend_len up to request r
    size_t c = 0;           // chunk pointer (advances monotonically)

    while (r < batch_size && c < num_chunks) {
      const int chunk_start = token_starts_ptr[c];
      const int chunk_end = token_starts_ptr[c + 1];
      const int extend_len = static_cast<int>(reqs[r]->GetInputIdsLength());
      const int seq_len = static_cast<int>(reqs[r]->forwarding_tokens->size());
      const int prefix_len = seq_len - extend_len;
      const int req_end = cu_extend_len + extend_len;

      // Chunk: [chunk_start, chunk_ned), Req: [cu_extend_len, req_end)
      const int overlap_start = std::max(cu_extend_len, chunk_start);
      const int overlap_end = std::min(req_end, chunk_end);

      if (overlap_start < overlap_end) {
        for (int i = overlap_start - cu_extend_len + 1; i <= overlap_end - cu_extend_len; ++i) {
          input.cu_seq_len_start_host.push_back(0);
          input.cu_seq_len_end_host.push_back((prefix_len + i) / compress_ratio_);
        }
      }

      if (req_end <= chunk_end) {
        // Request fully consumed within this chunk, advance to next request.
        cu_extend_len += extend_len;
        r++;
      } else {
        // Chunk fully consumed, advance to next chunk and reset KV accumulator.
        c++;
      }
    }
  } else {
    // Paged (decode) path: each request sees its own KV independently (start=0).
    for (const auto* req : input.dp_reqs) {
      const int seq_len = static_cast<int>(req->forwarding_tokens->size());
      const int extend_len = static_cast<int>(req->GetInputIdsLength());
      const int prefix_len = seq_len - extend_len;

      for (int i = 1; i <= extend_len; ++i) {
        input.cu_seq_len_start_host.push_back(0);
        input.cu_seq_len_end_host.push_back((prefix_len + i) / compress_ratio_);
      }
    }
  }

  HOST_TO_SHARED_DEVICE(cu_seq_len_start, input);
  HOST_TO_SHARED_DEVICE(cu_seq_len_end, input);
}

void IndexerMeta::PrepareBatchIndices(input_meta& input) {
  // Build per-token batch index array.
  // batch_indices[token_i] = batch_id (i.e., the request index).
  const auto& reqs = input.dp_reqs;
  const int batch_size = reqs.size();

  input.batch_indices_host.clear();
  input.batch_indices_host.reserve(input.total_dp_input_ids_len);

  for (int r = 0; r < batch_size; ++r) {
    const int extend_len = static_cast<int>(reqs[r]->GetInputIdsLength());
    for (int i = 0; i < extend_len; ++i) {
      input.batch_indices_host.push_back(r);
    }
  }

  HOST_TO_SHARED_DEVICE(batch_indices, input);
}

void IndexerMeta::PrepareSeqLen(input_meta& input) {
  // Build expanded context length for DeepGEMM paged path (q_seq_len > 1).
  // each element = context_len at that position.
  const auto& reqs = input.dp_reqs;
  const size_t q_seq_len = input.q_seq_len;

  input.expanded_seq_len_host.clear();
  input.expanded_seq_len_host.reserve(reqs.size() * q_seq_len);

  for (const auto* req : reqs) {
    for (size_t j = 1; j <= q_seq_len; ++j) {
      input.expanded_seq_len_host.push_back(req->kv_cached_token_num + j);
    }
  }

  HOST_TO_SHARED_DEVICE(expanded_seq_len, input);
}

}  // namespace ksana_llm
