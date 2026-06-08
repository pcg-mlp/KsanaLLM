/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"
#ifdef ENABLE_ACL
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#endif

namespace ksana_llm {

class HcclAllGatherLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  Status Forward(const Tensor* local, const Tensor* internal_buf, Tensor* output,
                 const LayerExecContext& exec_ctx = {});

 protected:
#ifdef ENABLE_ACL
  llm_kernels::utils::ATBOperationExecutor atb_all_gather_executor_;
  llm_kernels::utils::ATBOperationExecutor atb_permute_executor_;
#endif  // ENABLE_ACL
};

}  // namespace ksana_llm
