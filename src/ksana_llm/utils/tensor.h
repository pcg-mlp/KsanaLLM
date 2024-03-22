/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/common_tensor.h"
#include "ksana_llm/utils/device_types.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/utils/nvidia/nvidia_tensor.h"
#endif

#ifdef ENABLE_ACL
#  include "ksana_llm/utils/ascend/ascend_tensor.h"
#endif

namespace ksana_llm {

// The context for different device type.
using Tensor = TensorT<ACTIVE_DEVICE_TYPE>;

Status DestroyTensor(Tensor& tensor, const int rank);

Status CreateTensor(Tensor& tensor, const std::vector<size_t> shape, const DataType dtype, const int rank,
                    const MemoryDevice memory_device);

}  // namespace ksana_llm
