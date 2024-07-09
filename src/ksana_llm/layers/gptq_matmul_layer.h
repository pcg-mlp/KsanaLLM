/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

template <typename T, DataType WT>
class GPTQMatMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual int GetWorkSpaceSize() override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  size_t max_m, max_n, max_k;
  size_t groupsize;
};

}  // namespace ksana_llm
