/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/attention_layer.h"
#include "layers/dsv4_attention_types.h"

namespace ksana_llm {

// Flash attention layer for DeepSeek-V4.
// Handles prefill (context) phase with SWA (Sliding Window Attention) +
// optional extra (C4/C128 compressed paths) K cache.
class FlashDsv4AttentionLayer : public AttentionLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  Status Forward(const Dsv4AttentionInputs& inputs, Tensor* output);

  virtual size_t GetWorkspaceSize() override;

 private:
  template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
  Status ForwardT(const Dsv4AttentionInputs& inputs, Tensor* output);

  size_t max_token_num_ = 0;
  uint32_t page_bits_;              // log2(block_token_num)
  int compressed_block_token_num_;  // block_token_num / compress_ratio
};

}  // namespace ksana_llm
