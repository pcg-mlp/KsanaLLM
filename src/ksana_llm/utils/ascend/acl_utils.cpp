/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

std::vector<int64_t> GetAclTensorShape(aclTensor* tensor) {
  int64_t* tensor_shape_ptr = nullptr;
  uint64_t tensor_dims_num = 0;
  ACL_CHECK(aclGetViewShape(tensor, &tensor_shape_ptr, &tensor_dims_num));
  std::vector<int64_t> tensor_shape(tensor_dims_num);
  for (uint64_t i = 0; i < tensor_dims_num; ++i) {
    tensor_shape[i] = tensor_shape_ptr[i];
  }
  return tensor_shape;
}

DataType GetAclTensorDataType(aclTensor* tensor) {
  aclDataType data_type;
  ACL_CHECK(aclGetDataType(tensor, &data_type));
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

}  // namespace ksana_llm