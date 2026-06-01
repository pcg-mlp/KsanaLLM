/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "configure/context.h"
#include "configure/environment.h"
#include "models/base/forward_meta/indexer_meta.h"
#include "runtime/forward_request.h"
#include "utils/tensor.h"

namespace ksana_llm {

class C4IndexerMeta {
 public:
  C4IndexerMeta(const RuntimeConfig& runtime_config, int rank, std::shared_ptr<Context> context);

  void Prepare(const std::vector<ForwardRequest>& forward_reqs);

  struct input_meta : public IndexerMeta::input_meta {
    // C4-specific tensors (views of shared tensors)
    Tensor c4_seq_lens_tensor;
    Tensor expanded_c4_seq_lens_tensor;  // seq_lens after expansion of q_seq_len
    Tensor extend_lens_tensor;

    // C4-specific host vectors
    std::vector<int> c4_seq_lens_host;
    std::vector<int> expanded_c4_seq_lens_host;
    std::vector<int> extend_lens_host;

    size_t total_dp_compressed_input_ids_len = 0;

    void Reset() {
      IndexerMeta::input_meta::Reset();
      c4_seq_lens_host.clear();
      expanded_c4_seq_lens_host.clear();
      extend_lens_host.clear();
      total_dp_compressed_input_ids_len = 0;
    }
  };

  // Category is tightly coupled to underlying kernel
  input_meta flash_meta;               // extend_len is non-specialized, use flash attention
  std::vector<input_meta> page_metas;  // extend_len is fixed (`[1, decode_token_num_threshold]`), use page attention

 private:
  // Indexer kernel supports q_seq_len = 1/2 in decode phase
  // We use expand trick to handle q_seq_len > 2
  ForwardRequestType GetForwardRequestType(const ForwardRequest& forward_request) const {
    if (forward_request.GetInputIdsLength() <= GetDecodeTokenNumThreshold() &&
        forward_request.kv_cached_token_num > 0) {
      return ForwardRequestType::kPage;
    } else {
      return ForwardRequestType::kFlash;
    }
  }

  void PrepareC4SeqLens(input_meta& input);
  void PrepareExtendLens(std::vector<int>& host, const ForwardRequest& req);
  void PrepareBlockTable(input_meta& input);

  std::shared_ptr<Context> context_;

  uint32_t attn_dp_group_id_ = 0;

  static constexpr size_t kC4Ratio = 4;

  // Tensors shared by all input_metas
  Tensor c4_seq_lens_tensor;
  Tensor expanded_c4_seq_lens_tensor;
  Tensor extend_lens_tensor;

 public:
  // Number of dp tokens in context/decode phase
  size_t dp_context_tokens = 0;
  size_t dp_decode_tokens = 0;
  size_t dp_context_kv_tokens = 0;

  // Dummy tensor for indexer interface,
  // but the module parses input tensors positionally so we need a valid placeholder
  Tensor gathered_c4_out_loc_tensor;

  // Internal IndexerMeta (compress_ratio-aware)
  std::unique_ptr<IndexerMeta> indexer_meta;
};

}  // namespace ksana_llm
