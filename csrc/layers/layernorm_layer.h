/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"
#ifdef ENABLE_ACL
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#endif

namespace ksana_llm {

// The layernorm type.
enum class NormType { LAYERNORM = 0, RMSNORM = 1 };

class LayernormLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  // No-weight per-head RMSNorm.
  Status Forward(const Tensor* input, Tensor* output, const LayerExecContext& exec_ctx = {});
  // Single-path: RMSNorm (bias==nullptr) or LayerNorm (bias!=nullptr).
  Status Forward(const Tensor* input, const Tensor* weight, Tensor* output, const Tensor* bias = nullptr,
                 const LayerExecContext& exec_ctx = {});
  // Parallel two-path norm. stride1/stride2==0 means use hidden_size as row stride.
  Status Forward(const Tensor* input1, const Tensor* weight1, const Tensor* input2, const Tensor* weight2,
                 Tensor* output1, Tensor* output2, int stride1 = 0, int stride2 = 0,
                 const LayerExecContext& exec_ctx = {});

 private:
  template <typename T>
  Status ForwardNoWeightT(const Tensor& input, Tensor& output);
  template <typename T>
  Status ForwardWeightT(const Tensor& input, const Tensor& weight, Tensor& output, const Tensor* bias);
  template <typename T>
  Status ForwardParallelT(const Tensor& in1, const Tensor& w1, const Tensor& in2, const Tensor& w2, Tensor& out1,
                          Tensor& out2, int stride1, int stride2);

 protected:
  float rms_norm_eps_;
  // Used by per-head no-weight RMSNorm
  uint32_t num_heads_{0u};
  uint32_t head_dim_{0u};
  // NOTE(karlluo): only need by ascend
  int workspace_block_id_{-1};
  size_t workspace_size_{0ul};

 private:
  NormType norm_type_ = NormType::RMSNORM;

#ifdef ENABLE_ACL
  llm_kernels::utils::ATBOperationExecutor atb_op_executor_;
#endif  // ENABLE_ACL
};

}  // namespace ksana_llm
