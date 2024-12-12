/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/common_tensor.h"

namespace ksana_llm {

template <>
struct DeviceTensorTypeTraits<DEVICE_TYPE_ZIXIAO> {
  typedef void* value_type;
};

template <>
void TensorT<DEVICE_TYPE_ZIXIAO>::InitializeDeviceTensor();

template <>
void* TensorT<DEVICE_TYPE_ZIXIAO>::GetDeviceTensor();

template <>
std::vector<int64_t> TensorT<DEVICE_TYPE_ZIXIAO>::GetDeviceTensorShape() const;

template <>
DataType TensorT<DEVICE_TYPE_ZIXIAO>::GetDeviceTensorDataType() const;

template <>
void TensorT<DEVICE_TYPE_ZIXIAO>::ResetDeviceTensor(void* device_tensor);

}  // namespace ksana_llm
