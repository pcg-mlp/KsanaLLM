/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/add_layer.h"
#include "layers/base_layer.h"
#include "layers/layernorm_layer.h"
#ifdef ENABLE_ACL
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#endif

namespace ksana_llm {

class AddNormLayer : public BaseLayer {
 public:
  virtual ~AddNormLayer() = default;

  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  // input is normed in-place; residual is updated in-place as residual += input before norm.
  Status Forward(const Tensor* input, const Tensor* residual, const Tensor* weight, Tensor* output,
                 const LayerExecContext& exec_ctx = {});

 private:
  template <typename T>
  Status ForwardT(const Tensor* input, const Tensor* residual, const Tensor* weight, Tensor* output);

 protected:
  float rms_norm_eps_;
  std::shared_ptr<LayernormLayer> layernorm_layer_;
  std::shared_ptr<AddLayer> add_layer_;
};

}  // namespace ksana_llm
