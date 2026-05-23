/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/attention_layer.h"

namespace ksana_llm {

// Paged attention layer for DeepSeek-V4.
// Handles decode phase with SWA (Sliding Window Attention) +
// optional extra (C4/C128 compressed paths) K cache.
class PagedDsv4AttentionLayer : public AttentionLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  uint32_t page_bits_;              // log2(block_token_num)
  int compressed_block_token_num_;  // block_token_num / compress_ratio
};

}  // namespace ksana_llm
