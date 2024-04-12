/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/ascend/ascend_tensor.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

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

template <>
aclTensor* TensorT<DEVICE_TYPE_ASCEND>::ResetDeviceTensor(const DataType new_dtype,
                                                          const std::vector<int64_t> new_shape) {
  std::vector<int64_t> new_strides(new_shape.size(), 1);
  for (int64_t i = new_shape.size() - 2; i >= 0; i--) {
    new_strides[i] = new_shape[i + 1] * new_strides[i + 1];
  }

  void* input_dev_addr = GetPtr<void>();
  aclDestroyTensor(device_tensor_);
  device_tensor_ =
      aclCreateTensor(new_shape.data(), new_shape.size(), static_cast<aclDataType>(new_dtype), new_strides.data(), 0,
                      static_cast<aclFormat>(data_format), new_shape.data(), new_shape.size(), input_dev_addr);
  return device_tensor_;
}

void PrintAclTensorMeta(const aclTensor* tensor, const std::string& name) {
  int64_t* storageDims = nullptr;
  uint64_t storageDimsNum;
  ACL_CHECK(aclGetViewShape(tensor, &storageDims, &storageDimsNum));

  aclDataType dataType;
  ACL_CHECK(aclGetDataType(tensor, &dataType));

  aclFormat format;
  ACL_CHECK(aclGetFormat(tensor, &format));

  std::cout << name.c_str() << " dtype:" << dataType << ", shape:[";
  for (size_t i = 0; i < storageDimsNum; ++i) {
    int64_t dim = *(storageDims + i);
    if (i == 0) {
      std::cout << dim;
    } else {
      std::cout << ", " << dim;
    }
  }
  std::cout << "], format:" << format << std::endl;
}

}  // namespace ksana_llm
