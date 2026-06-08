/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/gather_first_dim_layer.h"

namespace ksana_llm {

Status GatherFirstDimLayer::Forward(const Tensor* data, const Tensor* indices, Tensor* output,
                                    const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "GatherFirstDimLayer not supported.");
}

}  // namespace ksana_llm
