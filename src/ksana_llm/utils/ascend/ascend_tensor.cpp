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
  ACL_CHECK(aclDestroyTensor(device_tensor_));
  device_tensor_ =
      aclCreateTensor(new_shape.data(), new_shape.size(), static_cast<aclDataType>(new_dtype), new_strides.data(), 0,
                      static_cast<aclFormat>(data_format), new_shape.data(), new_shape.size(), input_dev_addr);
  return device_tensor_;
}

void PrintAclTensorMeta(const aclTensor* tensor, const std::string& name) {
  int64_t* storage_dims = nullptr;
  uint64_t storage_dims_num;
  ACL_CHECK(aclGetViewShape(tensor, &storage_dims, &storage_dims_num));

  aclDataType data_type;
  ACL_CHECK(aclGetDataType(tensor, &data_type));

  aclFormat format;
  ACL_CHECK(aclGetFormat(tensor, &format));

  std::cout << name.c_str() << " dtype:" << data_type << ", shape:[";
  for (size_t i = 0; i < storage_dims_num; ++i) {
    int64_t dim = *(storage_dims + i);
    if (i == 0) {
      std::cout << dim;
    } else {
      std::cout << ", " << dim;
    }
  }
  std::cout << "], format:" << format << std::endl;
}

template <>
std::vector<int64_t> TensorT<DEVICE_TYPE_ASCEND>::GetDeviceTensorShape() const {
  int64_t* storage_dims = nullptr;
  uint64_t storage_dims_num;
  ACL_CHECK(aclGetViewShape(device_tensor_, &storage_dims, &storage_dims_num));
  std::vector<int64_t> device_tensor_shape;
  for (size_t i = 0; i < storage_dims_num; ++i) {
    device_tensor_shape.push_back(*(storage_dims + i));
  }
  return device_tensor_shape;
}

template <>
DataType TensorT<DEVICE_TYPE_ASCEND>::GetDeviceTensorDataType() const {
  aclDataType data_type;
  ACL_CHECK(aclGetDataType(device_tensor_, &data_type));
  switch (data_type) {
    case aclDataType::ACL_BF16:
      return DataType::TYPE_BF16;
    case aclDataType::ACL_BOOL:
      return DataType::TYPE_BOOL;
    case aclDataType::ACL_UINT8:
      return DataType::TYPE_UINT8;
    case aclDataType::ACL_UINT16:
      return DataType::TYPE_UINT16;
    case aclDataType::ACL_UINT32:
      return DataType::TYPE_UINT32;
    case aclDataType::ACL_UINT64:
      return DataType::TYPE_UINT64;
    case aclDataType::ACL_INT8:
      return DataType::TYPE_INT8;
    case aclDataType::ACL_INT16:
      return DataType::TYPE_INT16;
    case aclDataType::ACL_INT32:
      return DataType::TYPE_INT32;
    case aclDataType::ACL_INT64:
      return DataType::TYPE_INT64;
    case aclDataType::ACL_FLOAT16:
      return DataType::TYPE_FP16;
    case aclDataType::ACL_FLOAT:
      return DataType::TYPE_FP32;
    case aclDataType::ACL_DOUBLE:
      return DataType::TYPE_FP64;
    default:
      return DataType::TYPE_INVALID;
  }
}

template <>
void TensorT<DEVICE_TYPE_ASCEND>::ResetDeviceTensor(aclTensor* device_tensor) {
  device_tensor_ = device_tensor;
}

}  // namespace ksana_llm
