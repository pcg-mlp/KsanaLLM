/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/ascend/ascend_tensor.h"

namespace ksana_llm {

// Get the acl data type.
aclDataType GetAclDataType(DataType dtype) {
  static const std::unordered_map<DataType, aclDataType> dtype_to_acltype{{TYPE_INVALID, ACL_DT_UNDEFINED},
                                                                          {TYPE_BOOL, ACL_BOOL},
                                                                          {TYPE_UINT8, ACL_INT8},
                                                                          {TYPE_UINT16, ACL_UINT16},
                                                                          {TYPE_UINT32, ACL_UINT32},
                                                                          {TYPE_UINT64, ACL_INT64},
                                                                          {TYPE_INT8, ACL_INT8},
                                                                          {TYPE_INT16, ACL_INT16},
                                                                          {TYPE_INT32, ACL_INT32},
                                                                          {TYPE_INT64, ACL_INT64},
                                                                          {TYPE_BF16, ACL_BF16},
                                                                          {TYPE_FP16, ACL_FLOAT16},
                                                                          {TYPE_FP32, ACL_FLOAT},
                                                                          {TYPE_FP64, ACL_DOUBLE},
                                                                          {TYPE_BYTES, ACL_STRING},
                                                                          {TYPE_FP8_E4M3, ACL_DT_UNDEFINED},
                                                                          {TYPE_VOID, ACL_DT_UNDEFINED},
                                                                          {TYPE_POINTER, ACL_DT_UNDEFINED}};
  return dtype_to_acltype.at(dtype);
}

template <>
void TensorT<DEVICE_TYPE_ASCEND>::InitializeDeviceTensor() {
  std::vector<int64_t> strides(shape.size(), 1);
  std::vector<int64_t> acl_type_shape(shape.size(), 0);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    acl_type_shape[i] = static_cast<int64_t>(shape[i]);
  }
  void* device_addr;
  GetBlockManager()->GetContiguousPtr(block_id, device_addr);
  device_tensor_ =
      aclCreateTensor(acl_type_shape.data(), acl_type_shape.size(), GetAclDataType(dtype), strides.data(), 0,
                      aclFormat::ACL_FORMAT_ND, acl_type_shape.data(), acl_type_shape.size(), device_addr);
}

template <>
aclTensor* TensorT<DEVICE_TYPE_ASCEND>::GetDeviceTensor() {
  return device_tensor_;
}

}  // namespace ksana_llm
