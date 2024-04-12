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

}  // namespace ksana_llm