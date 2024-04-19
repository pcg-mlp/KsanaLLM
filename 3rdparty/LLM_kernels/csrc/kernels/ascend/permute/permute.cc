/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/ascend/permute/permute.h"

#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

void Permute(const aclTensor* permute_input, void** permute_input_tensor_addr_ptr, aclTensor** permute_output,
             const std::vector<int64_t>& dims, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  int64_t* input_t_shape_ptr = nullptr;
  uint64_t input_t_dims_num = 0;
  ACL_CHECK_RET(aclGetViewShape(permute_input, &input_t_shape_ptr, &input_t_dims_num));
  std::vector<int64_t> input_t_shape(input_t_dims_num);
  for (uint64_t i = 0; i < input_t_dims_num; ++i) {
    input_t_shape[i] = input_t_shape_ptr[i];
  }
  std::vector<int64_t> input_t_strides;
  utils::CalShapeStrides(input_t_shape, input_t_strides);

  std::vector<int64_t> output_t_shape(input_t_dims_num, 0);
  std::vector<int64_t> output_t_strides(input_t_shape.size(), 1);
  std::copy(input_t_shape.begin(), input_t_shape.end(), output_t_shape.begin());
  for (uint64_t i = 0; i < dims.size(); ++i) {
    output_t_shape[i] = input_t_shape[dims[i]];
    output_t_strides[i] = input_t_strides[dims[i]];
  }
  aclDataType acl_dtype;
  ACL_CHECK_RET(aclGetDataType(permute_input, &acl_dtype));
  *permute_output = aclCreateTensor(output_t_shape.data(), output_t_shape.size(), acl_dtype, output_t_strides.data(), 0,
                                    aclFormat::ACL_FORMAT_ND, output_t_shape.data(), output_t_shape.size(),
                                    *permute_input_tensor_addr_ptr);

  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

}  // namespace ascend
}  // namespace llm_kernels
