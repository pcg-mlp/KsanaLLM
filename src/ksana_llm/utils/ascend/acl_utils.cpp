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

  static const std::unordered_map<aclDataType, DataType> acl_type_map{
      {aclDataType::ACL_BF16, DataType::TYPE_BF16},     {aclDataType::ACL_BOOL, DataType::TYPE_BOOL},
      {aclDataType::ACL_UINT8, DataType::TYPE_UINT8},   {aclDataType::ACL_UINT16, DataType::TYPE_UINT16},
      {aclDataType::ACL_UINT32, DataType::TYPE_UINT32}, {aclDataType::ACL_UINT64, DataType::TYPE_UINT64},
      {aclDataType::ACL_INT8, DataType::TYPE_INT8},     {aclDataType::ACL_INT16, DataType::TYPE_INT16},
      {aclDataType::ACL_INT32, DataType::TYPE_INT32},   {aclDataType::ACL_INT64, DataType::TYPE_INT64},
      {aclDataType::ACL_FLOAT16, DataType::TYPE_FP16},  {aclDataType::ACL_FLOAT, DataType::TYPE_FP32},
      {aclDataType::ACL_DOUBLE, DataType::TYPE_FP64}};
  return acl_type_map.count(data_type) ? acl_type_map.at(data_type) : DataType::TYPE_INVALID;
}

std::vector<int>& GetPaddedTokenSize() {
  static std::vector<int> s_padded_tokens;
  return s_padded_tokens;
}

}  // namespace ksana_llm
