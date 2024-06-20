/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

template <typename T>
class Fp8MatMulLayer : public BaseLayer {
 public:
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override {
    return Status(RET_RUNTIME, fmt::format("Fp8MatMulLayer forward is not supported\n"));
  }
};

}  // namespace ksana_llm
