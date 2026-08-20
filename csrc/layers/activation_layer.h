/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

enum class ActivationType {
  Gelu = 0,
  Relu = 1,
  Geglu = 2,
  Swiglu = 3,
};

class ActivationLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  // gated_weight/bias/gated_bias are optional (nullptr when unused).
  Status Forward(const Tensor* input, Tensor* output, const Tensor* gated_weight = nullptr,
                 const Tensor* bias = nullptr, const Tensor* gated_bias = nullptr,
                 const LayerExecContext& exec_ctx = {});

 protected:
  template <typename T>
  Status ForwardT(const Tensor* input, Tensor* output, const Tensor* gated_weight, const Tensor* bias,
                  const Tensor* gated_bias);

 protected:
  ActivationType activation_type_;
};

class SigmoidLayer : public BaseLayer {
 public:
  // In-place sigmoid on inout.
  Status Forward(Tensor* inout, const LayerExecContext& exec_ctx = {});

 private:
  template <typename T>
  Status ForwardT(Tensor* inout);
};

}  // namespace ksana_llm
