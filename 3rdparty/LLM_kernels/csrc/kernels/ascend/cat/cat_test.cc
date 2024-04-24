/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half.hpp"
#include "csrc/kernels/ascend/cat/cat.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendCatTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::context;
  using AscendTestSuitBase::default_device;
  using AscendTestSuitBase::is_inited;
  using AscendTestSuitBase::stream;
};

TEST_F(LlamaAscendCatTestSuit, CatTest) {
  const std::vector<int64_t> input_shape = {1, 2};
  aclTensor* input_tensor_1 = nullptr;
  void* input_workspace_1 = nullptr;
  aclTensor* input_tensor_2 = nullptr;
  void* input_workspace_2 = nullptr;

  const std::vector<int64_t> output_shape = {1, 4};
  aclTensor* output_tensor = nullptr;
  void* output_workspace = nullptr;

  CreateAclTensor(input_shape, &input_workspace_1, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &input_tensor_1);
  CreateAclTensor(input_shape, &input_workspace_2, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &input_tensor_2);
  CreateAclTensor(output_shape, &output_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &output_tensor);
  std::vector<half_float::half> input_vec_1_host(GetShapeSize(input_shape));
  std::vector<half_float::half> input_vec_2_host(GetShapeSize(input_shape));
  std::vector<half_float::half> out_vec_host(GetShapeSize(output_shape));
  for (size_t i = 0; i < input_vec_1_host.size(); ++i) {
    input_vec_1_host[i] = (half_float::half)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    input_vec_2_host[i] = (half_float::half)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  ACL_CHECK_RET(aclrtMemcpyAsync(input_workspace_1, GetShapeSize(input_shape) * sizeof(half_float::half),
                                 input_vec_1_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtMemcpyAsync(input_workspace_2, GetShapeSize(input_shape) * sizeof(half_float::half),
                                 input_vec_2_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  std::vector<const aclTensor*> inputs = {input_tensor_1, input_tensor_2};
  int64_t cat_dim = -1;

  Cat(inputs, cat_dim, &output_tensor, stream, llm_kernels::utils::GetTestWorkSpaceFunc);

  ACL_CHECK_RET(aclrtMemcpyAsync(out_vec_host.data(), GetShapeSize(output_shape) * sizeof(half_float::half),
                                 output_workspace, GetShapeSize(output_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  for (size_t i = 0; i < input_vec_1_host.size(); ++i) {
    EXPECT_NEAR(float(input_vec_1_host[i]), float(out_vec_host[i]), 1e-5);
  }
  for (size_t i = 0; i < input_vec_2_host.size(); ++i) {
    EXPECT_NEAR(float(input_vec_2_host[i]), float(out_vec_host[input_vec_1_host.size() + i]), 1e-5);
  }

  ACL_CHECK_RET(aclDestroyTensor(output_tensor));
  ACL_CHECK_RET(aclDestroyTensor(input_tensor_1));
  ACL_CHECK_RET(aclDestroyTensor(input_tensor_2));
  ACL_CHECK_RET(aclrtFree(input_workspace_1));
  ACL_CHECK_RET(aclrtFree(input_workspace_2));
  ACL_CHECK_RET(aclrtFree(output_workspace));
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
