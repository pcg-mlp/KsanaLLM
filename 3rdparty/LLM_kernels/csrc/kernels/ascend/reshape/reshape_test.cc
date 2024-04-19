/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "3rdparty/half.hpp"
#include "csrc/kernels/ascend/reshape/reshape.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

#include "aclnnop/aclnn_copy.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class AscendReshapeTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::stream;
};

TEST_F(AscendReshapeTestSuit, CommonTest) {
  aclDataType dtype = aclDataType::ACL_FLOAT16;
  aclTensor* input_tensor = nullptr;
  void* input_dev = nullptr;
  const std::vector<int64_t> input_shape = {1, 2, 15};
  CreateAclTensor(input_shape, &input_dev, dtype, aclFormat::ACL_FORMAT_ND, &input_tensor);

  const std::vector<int64_t> output_shape = {1, 2, 3, 5};
  aclTensor* output_tensor = nullptr;
  Reshape(input_tensor, &input_dev, output_shape, &output_tensor, stream, llm_kernels::utils::GetTestWorkSpaceFunc);

  aclDataType output_dtype;
  ACL_CHECK_RET(aclGetDataType(output_tensor, &output_dtype));
  EXPECT_EQ(output_dtype, dtype);

  aclFormat output_fmt;
  ACL_CHECK_RET(aclGetFormat(output_tensor, &output_fmt));
  EXPECT_EQ(output_fmt, aclFormat::ACL_FORMAT_ND);

  int64_t* output_t_shape_ptr = nullptr;
  uint64_t output_t_shape_num = 0;
  ACL_CHECK_RET(aclGetViewShape(output_tensor, &output_t_shape_ptr, &output_t_shape_num));
  EXPECT_EQ(output_shape.size(), output_t_shape_num);
  for (uint64_t i = 0; i < output_t_shape_num; ++i) {
    EXPECT_EQ(output_t_shape_ptr[i], output_shape[i]);
  }

  ACL_CHECK_RET(aclDestroyTensor(output_tensor));
  ACL_CHECK_RET(aclDestroyTensor(input_tensor));
  ACL_CHECK_RET(aclrtFree(input_dev));
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
