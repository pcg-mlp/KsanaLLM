/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/add_layer.h"

namespace ksana_llm {

Status AddLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // TODO(karlluo): implement llm_kernels::ascend::Add
  return Status();
}
}  // namespace ksana_llm
