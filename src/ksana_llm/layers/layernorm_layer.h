/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

template <typename T>
class LayernormLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 protected:
  float rms_norm_eps_;
  // NOTE(karlluo): only need by ascend
  int workspace_block_id_{-1};
  size_t workspace_size_{0ul};
};

}  // namespace ksana_llm
