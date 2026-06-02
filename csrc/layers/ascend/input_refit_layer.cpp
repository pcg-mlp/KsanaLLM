/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/input_refit_layer.h"

namespace ksana_llm {

Status InputRefitLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "InputRefitLayer not supported.");
}
}  // namespace ksana_llm
