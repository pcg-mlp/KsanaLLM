/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "kernels/trans_layout.h"

namespace ksana_llm {

Status TransLayout(Tensor& tensor, Stream& stream, void* device_ctx) {
  return Status(RET_UNDEFINED_REFERENCE, "TransLayout not yet supported on Kunlun.");
}

}  // namespace ksana_llm
