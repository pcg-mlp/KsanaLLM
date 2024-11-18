/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

enum class ActivationType {
  Gelu = 0,
  Relu = 1,
  Geglu = 2,
  Swiglu = 3,
};

template <ActivationType ACTIVATION_TYPE>
constexpr bool IsGatedActivation() {
  if constexpr (ACTIVATION_TYPE == ActivationType::Geglu || ACTIVATION_TYPE == ActivationType::Swiglu) {
    return true;
  }
  return false;
}

template <ActivationType ACTIVATION_TYPE, typename T>
class ActivationLayer : public BaseLayer {
 public:
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;
};

template <typename T>
class SigmoidLayer : public BaseLayer {
 public:
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;
};

}  // namespace ksana_llm
