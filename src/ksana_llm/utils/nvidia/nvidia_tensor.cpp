/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/nvidia/nvidia_tensor.h"

#include <type_traits>

namespace ksana_llm {

template <>
void TensorT<DEVICE_TYPE_NVIDIA>::InitializeDeviceTensor() {}

template <>
void* TensorT<DEVICE_TYPE_NVIDIA>::GetDeviceTensor() {
  return nullptr;
}

template half* TensorT<DEVICE_TYPE_NVIDIA>::GetPtr<half>() const;

#ifdef ENABLE_BFLOAT16
template __nv_bfloat16* TensorT<DEVICE_TYPE_NVIDIA>::GetPtr<__nv_bfloat16>() const;
#endif

#ifdef ENABLE_FP8
template __nv_fp8_e4m3* TensorT<DEVICE_TYPE_NVIDIA>::GetPtr<__nv_fp8_e4m3>() const;
#endif

template <>
std::vector<int64_t> TensorT<DEVICE_TYPE_NVIDIA>::GetDeviceTensorShape() const {
  std::vector<int64_t> device_tensor_shape;
  return device_tensor_shape;
}

template <>
DataType TensorT<DEVICE_TYPE_NVIDIA>::GetDeviceTensorDataType() const {
  return DataType::TYPE_INVALID;
}

template <>
void TensorT<DEVICE_TYPE_NVIDIA>::ResetDeviceTensor(void* device_tensor) {}

}  // namespace ksana_llm
