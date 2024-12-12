/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/zixiao/zixiao_tensor.h"

#include <type_traits>

namespace ksana_llm {

template <>
void TensorT<DEVICE_TYPE_ZIXIAO>::InitializeDeviceTensor() {}

template <>
void* TensorT<DEVICE_TYPE_ZIXIAO>::GetDeviceTensor() {
  return nullptr;
}

template float16* TensorT<DEVICE_TYPE_ZIXIAO>::GetPtr<float16>() const;

#ifdef ENABLE_BFLOAT16
template bfloat16* TensorT<DEVICE_TYPE_ZIXIAO>::GetPtr<bfloat16>() const;
#endif

template <>
std::vector<int64_t> TensorT<DEVICE_TYPE_ZIXIAO>::GetDeviceTensorShape() const {
  std::vector<int64_t> device_tensor_shape;
  return device_tensor_shape;
}

template <>
DataType TensorT<DEVICE_TYPE_ZIXIAO>::GetDeviceTensorDataType() const {
  return DataType::TYPE_INVALID;
}

template <>
void TensorT<DEVICE_TYPE_ZIXIAO>::ResetDeviceTensor(void* device_tensor) {}

}  // namespace ksana_llm
