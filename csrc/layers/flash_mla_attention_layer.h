/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/attention_layer.h"

namespace ksana_llm {

class FlashMlaAttentionLayer : public AttentionLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx = {}) override;

  virtual size_t GetWorkspaceSize() override;

 private:
  template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  size_t max_step_token_num_ = 0;
  size_t max_chunked_kv_tokens_ = 0;
};

}  // namespace ksana_llm
