/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/kernels/cast.h"

namespace ksana_llm {

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* workspace_ptr) {
  return Status(RET_UNDEFINED_REFERENCE, "CastInplace not supported.");
}

}  // namespace ksana_llm
