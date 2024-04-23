/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/common_tensor.h"

namespace ksana_llm {

template <>
struct DeviceTensorTypeTraits<DEVICE_TYPE_NVIDIA> {
    typedef void* value_type;
};

template <>
void TensorT<DEVICE_TYPE_NVIDIA>::InitializeDeviceTensor();

template <>
void* TensorT<DEVICE_TYPE_NVIDIA>::GetDeviceTensor();

template <>
std::vector<int64_t> TensorT<DEVICE_TYPE_NVIDIA>::GetDeviceTensorShape() const;

template <>
DataType TensorT<DEVICE_TYPE_NVIDIA>::GetDeviceTensorDataType() const;

template <>
void TensorT<DEVICE_TYPE_NVIDIA>::ResetDeviceTensor(void* device_tensor);

}  // namespace ksana_llm
