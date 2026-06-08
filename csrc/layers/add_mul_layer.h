/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

class AddMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  Status Forward(const Tensor* input1, const Tensor* input2, Tensor* output, const LayerExecContext& exec_ctx = {});

 private:
  template <typename T>
  Status ForwardT(const Tensor& input1, const Tensor& input2, Tensor& output);

 private:
  float scale1_{static_cast<float>(1.0f)};
  float scale2_{static_cast<float>(1.0f)};
};

}  // namespace ksana_llm
