/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/models/base/base_weight.h"

namespace ksana_llm {

template <typename T, DataType WT>
class GroupMatMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkSpaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config_) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  size_t max_m, max_n, max_k;
  size_t groupsize;

  std::map<std::array<size_t, 3>, size_t> config_map_;
};

}  // namespace ksana_llm
