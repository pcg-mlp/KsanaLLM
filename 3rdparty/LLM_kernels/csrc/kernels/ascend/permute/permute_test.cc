/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "3rdparty/half.hpp"
#include "csrc/kernels/ascend/permute/permute.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

#include "aclnnop/aclnn_copy.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendPermuteTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::stream;
};

TEST_F(LlamaAscendPermuteTestSuit, CommonTest) {
  aclDataType dtype = aclDataType::ACL_FLOAT16;
  uint64_t workspace_size = 0ul;
  void* workspace_ptr = nullptr;

  aclTensor* input_tensor = nullptr;
  void* input_workspace = nullptr;
  const std::vector<int64_t> input_shape = {3, 4, 5};
  CreateAclTensor(input_shape, &input_workspace, dtype, aclFormat::ACL_FORMAT_ND, &input_tensor);

  const std::vector<int64_t> dims = {2, 0, 1};
  aclTensor* output_tensor = nullptr;
  Permute(input_tensor, &input_workspace, &output_tensor, dims, stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  aclDataType output_dtype;
  ACL_CHECK_RET(aclGetDataType(output_tensor, &output_dtype));
  EXPECT_EQ(output_dtype, dtype);

  int64_t* output_t_shape_ptr = nullptr;
  uint64_t output_t_dims_num = 0;
  ACL_CHECK_RET(aclGetViewShape(output_tensor, &output_t_shape_ptr, &output_t_dims_num));
  for (uint64_t i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(output_t_shape_ptr[i], input_shape[dims[i]]);
  }

  if (workspace_size > 0 && workspace_ptr != nullptr) {
    ACL_CHECK_RET(aclrtFree(workspace_ptr));
  }
  ACL_CHECK_RET(aclDestroyTensor(output_tensor));
  ACL_CHECK_RET(aclDestroyTensor(input_tensor));
  ACL_CHECK_RET(aclrtFree(input_workspace));
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
