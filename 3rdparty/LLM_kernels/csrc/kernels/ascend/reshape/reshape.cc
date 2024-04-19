/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "reshape.h"

#include <vector>

#include "aclnnop/aclnn_copy.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

void Reshape(const aclTensor* input, void** inputDev, const std::vector<int64_t>& output_t_shape, aclTensor** output,
             aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  // do not copy only change shape and stride
  int64_t* input_shape = nullptr;
  uint64_t input_shape_num = 0;
  ACL_CHECK_RET(aclGetViewShape(input, &input_shape, &input_shape_num));
  std::vector<int64_t> input_t_shape(input_shape_num);
  uint64_t input_elementsize = 1;
  for (uint64_t i = 0; i < input_shape_num; ++i) {
    input_elementsize *= input_shape[i];
    input_t_shape[i] = input_shape[i];
  }
  std::vector<int64_t> input_t_strides;
  utils::CalShapeStrides(input_t_shape, input_t_strides);

  ACL_CHECK_EQ(input_elementsize, utils::GetShapeSize(output_t_shape));
  std::vector<int64_t> output_t_strides;
  utils::CalShapeStrides(output_t_shape, output_t_strides);

  aclDataType acl_dtype;
  ACL_CHECK_RET(aclGetDataType(input, &acl_dtype));
  aclFormat acl_fmt;
  ACL_CHECK_RET(aclGetFormat(input, &acl_fmt));
  *output = aclCreateTensor(output_t_shape.data(), output_t_shape.size(), acl_dtype, output_t_strides.data(), 0,
                            acl_fmt, output_t_shape.data(), output_t_shape.size(), *inputDev);
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

}  // namespace ascend
}  // namespace llm_kernels
