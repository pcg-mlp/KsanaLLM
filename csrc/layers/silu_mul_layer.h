/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"
#ifdef ENABLE_ACL
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#endif

namespace ksana_llm {

class SiluMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  // Fused gate_up input: silu(gate) * up in a single [token_num, 2*hidden] tensor.
  Status Forward(const Tensor* fused_gate_up, Tensor* output, const LayerExecContext& exec_ctx = {});
  // Split gate and up inputs separately.
  Status Forward(const Tensor* gate, const Tensor* up, Tensor* output, const LayerExecContext& exec_ctx = {});

 private:
  template <typename T>
  Status ForwardFusedT(const Tensor& fused_gate_up, Tensor& output, const LayerExecContext& exec_ctx);
  template <typename T>
  Status ForwardSplitT(const Tensor& gate, const Tensor& up, Tensor& output, const LayerExecContext& exec_ctx);

  float swiglu_limit_;

 protected:
#ifdef ENABLE_ACL
  llm_kernels::utils::ATBOperationExecutor atb_op_executor_;
#endif  // ENABLE_ACL
};

}  // namespace ksana_llm
