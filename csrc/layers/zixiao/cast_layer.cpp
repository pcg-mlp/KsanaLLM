/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/cast_layer.h"

namespace ksana_llm {

Status CastLayer::Forward(const Tensor* input, size_t byte_offset, size_t vocab_size, size_t vocab_size_pad,
                          Tensor* output, const LayerExecContext& exec_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "CastLayer not supported.");
}

}  // namespace ksana_llm
