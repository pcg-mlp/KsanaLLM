/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/common_tensor.h"
#include "ksana_llm/utils/device_types.h"

namespace ksana_llm {

template <>
struct DeviceTensorTypeTraits<DEVICE_TYPE_NVIDIA> {
  typedef void* value_type;
};

template <>
void TensorT<DEVICE_TYPE_NVIDIA>::InitializeDeviceTensor();

template <>
void* TensorT<DEVICE_TYPE_NVIDIA>::GetDeviceTensor();

}  // namespace ksana_llm
