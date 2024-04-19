/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <any>

#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* workspace_ptr = nullptr);

}  // namespace ksana_llm
