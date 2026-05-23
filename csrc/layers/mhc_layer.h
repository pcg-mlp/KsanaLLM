/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"
#ifdef ENABLE_ACL
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#endif

namespace ksana_llm {

class MhcLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

  // The num_splits used by mhc_pre kernel
  size_t GetMhcPreNumSplits(int token_num, int hc_hidden_dim);

  static constexpr size_t kMaxNumSplits = 64;

 private:
  // Down-project input before attn/mlp
  template <typename T>
  Status PreForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // Down-project input before attn/mlp, with fused RMSNorm
  template <typename T>
  Status PreForwardWithNormT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // Combine input and residual after attn/mlp
  template <typename T>
  Status PostForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // Up-project input after embedding lookup
  template <typename T>
  Status InitialForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // Down-project input before lmhead
  template <typename T>
  Status FinalForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  float rms_norm_eps_;
  uint32_t hc_mult_;
  uint32_t hc_sinkhorn_iters_;
  float hc_eps_;
  size_t hidden_dim_;
  size_t max_token_num_;

#ifdef ENABLE_ACL
  llm_kernels::utils::ATBOperationExecutor atb_op_executor_;
#endif  // ENABLE_ACL
};

}  // namespace ksana_llm
