/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "configure/context.h"
#include "configure/environment.h"
#include "runtime/forward_request.h"
#include "utils/tensor.h"

namespace ksana_llm {

class IndexerMeta {
  friend class C4IndexerMeta;

 public:
  IndexerMeta(const RuntimeConfig& runtime_config, int rank, std::shared_ptr<Context> context, int compress_ratio = 1);

  void Prepare(const std::vector<ForwardRequest>& forward_reqs);

  struct input_info {
    std::vector<const ForwardRequest*> dp_reqs;

    Tensor cu_seq_len_start_tensor;
    Tensor cu_seq_len_end_tensor;
    // Per-token batch index: batch_indices[t] = which request (batch) token t belongs to.
    Tensor batch_indices_tensor;

    // Persistent host vectors to prevent destruction during MemcpyAsync
    std::vector<int> cu_seq_len_start_host;
    std::vector<int> cu_seq_len_end_host;
    std::vector<int> batch_indices_host;

    size_t q_seq_len = 0;
    size_t total_dp_input_ids_len = 0;

    void Reset() {
      dp_reqs.clear();
      cu_seq_len_start_host.clear();
      cu_seq_len_end_host.clear();
      batch_indices_host.clear();
      q_seq_len = 0;
      total_dp_input_ids_len = 0;
    }
  };

  // Category is tightly coupled to underlying kernel
  input_info flash_input;               // extend_len is non-specialized, use flash attention
  std::vector<input_info> page_inputs;  // extend_len is fixed (`[1, decode_token_num_threshold]`), use page attention

  // Number of dp tokens in context/decode phase
  size_t dp_context_tokens = 0;
  size_t dp_decode_tokens = 0;

  // Offsets for chunked indexer: reqs are split into chunks so that each chunk's total Q tokens
  // stay within `max_chunked_q_tokens_` AND total KV tokens stay within `max_chunked_kv_tokens_`.
  Tensor chunk_token_starts_host;     // HOST, INT32, [num_chunks + 1], Q token boundaries (with sentinel, consecutive)
  Tensor chunk_kv_token_ranges_host;  // HOST, INT32, [2 * num_chunks], KV [start, end) per chunk (pairs, may overlap)

 private:
  ForwardRequestType GetForwardRequestType(const ForwardRequest& forward_request) const {
    if (forward_request.GetInputIdsLength() <= GetDecodeTokenNumThreshold() &&
        forward_request.kv_cached_token_num > 0) {
      return ForwardRequestType::kPage;
    } else {
      return ForwardRequestType::kFlash;
    }
  }

  // Split flash reqs into chunks bounded by both `max_chunked_q_tokens_` and `max_chunked_kv_tokens_`,
  // taking the tighter constraint, and build per-chunk offset tensors.
  void PrepareChunk(input_info& input);
  // Build per-token batch index array for the given input (replaces extend_lens scan in topk kernel).
  void PrepareBatchIndices(input_info& input);
  // Build per Q-token KV start/end. Flash path uses chunk boundaries; paged path uses per-request isolation.
  void PrepareCuSeqLen(input_info& input, bool is_paged = false);

  std::shared_ptr<Context> context_;

  uint32_t attn_dp_group_id_ = 0;

  // Chunk limits
  int max_chunked_q_tokens_ = 0;   // Computed from 4GB logits constraint + env override
  int max_chunked_kv_tokens_ = 0;  // From runtime config
  int compress_ratio_ = 1;         // KV compression ratio: 1 for DSA, 4 for C4

  // Tensors shared by all input_infos
  // Per Q-token KV range: [start[t], end[t]) defines the KV window for token t.
  Tensor cu_seq_len_start_tensor;  // DEVICE, INT32, [max_token_num]
  Tensor cu_seq_len_end_tensor;    // DEVICE, INT32, [max_token_num]
  // Per-token batch index: batch_indices[t] = which request (batch) token t belongs to.
  Tensor batch_indices_tensor;  // DEVICE, INT32, [max_token_num]
};

}  // namespace ksana_llm
