/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"
#ifdef ENABLE_ACL
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#endif

namespace ksana_llm {

class AddLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  // Typed interface: a and b are added element-wise into output.
  // b may have a different shape[0] than a to trigger broadcast (bias-add mode).
  Status Forward(const Tensor* a, const Tensor* b, Tensor* output, const LayerExecContext& exec_ctx = {});

 private:
  template <typename T>
  Status ForwardT(const Tensor* a, const Tensor* b, Tensor* output);

 private:
#ifdef ENABLE_ACL
  llm_kernels::utils::ATBOperationExecutor atb_op_executor_;
#endif  // ENABLE_ACL
};

}  // namespace ksana_llm
