/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/ascend/ascend_tensor.h"

namespace ksana_llm {

template <>
void TensorT<DEVICE_TYPE_ASCEND>::InitializeDeviceTensor() {
  if (strides.empty()) {
    strides.resize(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
  }

  std::vector<int64_t> acl_type_shape(shape.size(), 0);
  for (size_t i = 0; i < shape.size(); ++i) {
    acl_type_shape[i] = static_cast<int64_t>(shape[i]);
  }
  void* device_addr;
  GetBlockManager()->GetContiguousPtr(block_id, device_addr);
  device_tensor_ =
      aclCreateTensor(acl_type_shape.data(), acl_type_shape.size(), static_cast<aclDataType>(dtype), strides.data(), 0,
                      static_cast<aclFormat>(data_format), acl_type_shape.data(), acl_type_shape.size(), device_addr);
}

template <>
aclTensor* TensorT<DEVICE_TYPE_ASCEND>::GetDeviceTensor() {
  return device_tensor_;
}

}  // namespace ksana_llm
