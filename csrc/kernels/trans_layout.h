/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "device/device_utils.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

// NOTE(karlluo): for some case: matmul, different layout has much performance disparity. For example, on Ascend NPU, NZ
// is better than ND. TransLayout is such operation to transform layout from ND to NZ. In the meanwhile, on NVIDIA GPU,
// TransLayout is a conversion function between col-major to row-major.
Status TransLayout(Tensor& tensor, Stream& stream, void* device_ctx = nullptr);

}  // namespace ksana_llm
