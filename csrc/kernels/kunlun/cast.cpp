/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "kernels/cast.h"

namespace ksana_llm {

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* device_ctx, void* workspace_ptr) {
  return Status(RET_UNDEFINED_REFERENCE, "CastInplace not yet supported on Kunlun.");
}

}  // namespace ksana_llm
