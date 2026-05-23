/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/gather_first_dim_layer.h"

namespace ksana_llm {

Status GatherFirstDimLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "GatherFirstDimLayer not supported.");
}

}  // namespace ksana_llm
