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

class MlaMeta {
 public:
  MlaMeta(const RuntimeConfig& runtime_config, int rank, std::shared_ptr<Context> context);

  void Prepare(const std::vector<ForwardRequest>& forward_reqs);

  struct input_info {
    std::vector<const ForwardRequest*> dp_reqs;

    void Reset() { dp_reqs.clear(); }
  };

  // Category is tightly coupled to underlying kernel
  input_info flash_input;  // extend_len is non-specialized, use flash attention

  // Number of dp tokens in context/decode phase
  size_t dp_context_tokens = 0;

  // Offsets for chunked KV: reqs are split into chunks so that each chunk's total tokens
  // stay within `max_chunked_kv_tokens`.
  Tensor chunk_kv_batch_starts_host;      // HOST, INT32, [num_chunks + 1], req index where each chunk begins
  Tensor chunk_kv_cu_extend_lens_tensor;  // DEVICE, INT32, [batch_size + num_chunks], cumulative extend lens per chunk
  Tensor chunk_kv_cu_seq_lens_tensor;     // DEVICE, INT32, [batch_size + num_chunks], cumulative seq lens per chunk

 private:
  ForwardRequestType GetForwardRequestType(const ForwardRequest& forward_request) const {
    if (forward_request.GetInputIdsLength() <= GetDecodeTokenNumThreshold() &&
        forward_request.kv_cached_token_num > 0) {
      return ForwardRequestType::kPage;
    } else {
      return ForwardRequestType::kFlash;
    }
  }

  // Split reqs into chunks bounded by `max_chunked_kv_tokens` and build per-chunk offset tensors.
  void PrepareChunkKV();

  std::shared_ptr<Context> context_;

  uint32_t attn_dp_group_id_ = 0;

  int max_chunked_kv_tokens_ = 0;

  // Intermediate host vectors (persist during async memcpy)
  std::vector<int> chunk_kv_cu_extend_lens_list_;
  std::vector<int> chunk_kv_cu_seq_lens_list_;
};

}  // namespace ksana_llm
