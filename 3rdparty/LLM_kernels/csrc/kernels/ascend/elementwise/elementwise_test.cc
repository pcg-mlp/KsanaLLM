/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "csrc/kernels/ascend/elementwise/elementwise.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendElementwiseTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::context;
  using AscendTestSuitBase::default_device;
  using AscendTestSuitBase::is_inited;
  using AscendTestSuitBase::stream;
};

TEST_F(LlamaAscendElementwiseTestSuit, AddsTest) {
  aclTensor* input_tensor = nullptr;
  void* input_workspace = nullptr;
  const std::vector<int64_t> input_shape = {1, 2};
  aclTensor* other_tensor = nullptr;
  void* other_workspace = nullptr;

  aclTensor* output_tensor = nullptr;
  void* output_workspace = nullptr;
  const std::vector<int64_t> output_shape = {1, 2};
  CreateAclTensor(input_shape, &input_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &input_tensor);
  CreateAclTensor(input_shape, &other_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &other_tensor);
  CreateAclTensor(output_shape, &output_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &output_tensor);
  std::vector<half_float::half> input_vec_host(GetShapeSize(input_shape));
  std::vector<half_float::half> other_vec_host(GetShapeSize(input_shape));
  std::vector<half_float::half> out_vec_host(GetShapeSize(output_shape));
  for (size_t i = 0; i < input_vec_host.size(); ++i) {
    input_vec_host[i] = (half_float::half)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    other_vec_host[i] = (half_float::half)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  uint16_t one_in_fp16 = 0b11110000000000;
  aclScalar* add_alpha = aclCreateScalar(&one_in_fp16, aclDataType::ACL_FLOAT16);
  ACL_CHECK_RET(aclrtMemcpyAsync(input_workspace, GetShapeSize(input_shape) * sizeof(half_float::half),
                                 input_vec_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtMemcpyAsync(other_workspace, GetShapeSize(input_shape) * sizeof(half_float::half),
                                 other_vec_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  Adds(input_tensor, add_alpha, add_alpha, &output_tensor, stream, llm_kernels::utils::GetTestWorkSpaceFunc);

  ACL_CHECK_RET(aclrtMemcpyAsync(out_vec_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 output_workspace, GetShapeSize(output_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  for (size_t i = 0; i < input_vec_host.size(); ++i) {
    EXPECT_NEAR(float(input_vec_host[i]) + 1.0f * 1.0f, float(out_vec_host[i]), 1e-3);
  }

  ACL_CHECK_RET(aclDestroyTensor(output_tensor));
  ACL_CHECK_RET(aclDestroyTensor(input_tensor));
  ACL_CHECK_RET(aclDestroyTensor(other_tensor));
  ACL_CHECK_RET(aclrtFree(input_workspace));
  ACL_CHECK_RET(aclrtFree(output_workspace));
  ACL_CHECK_RET(aclrtFree(other_workspace));
}

TEST_F(LlamaAscendElementwiseTestSuit, MulTest) {
  aclTensor* input_tensor = nullptr;
  void* input_workspace = nullptr;
  const std::vector<int64_t> input_shape = {1, 2};
  aclTensor* other_tensor = nullptr;
  void* other_workspace = nullptr;

  aclTensor* output_tensor = nullptr;
  void* output_workspace = nullptr;
  const std::vector<int64_t> output_shape = {1, 2};
  CreateAclTensor(input_shape, &input_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &input_tensor);
  CreateAclTensor(input_shape, &other_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &other_tensor);
  CreateAclTensor(output_shape, &output_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &output_tensor);
  std::vector<half_float::half> input_vec_host(GetShapeSize(input_shape));
  std::vector<half_float::half> other_vec_host(GetShapeSize(input_shape));
  std::vector<half_float::half> out_vec_host(GetShapeSize(output_shape));
  for (size_t i = 0; i < input_vec_host.size(); ++i) {
    input_vec_host[i] = (half_float::half)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    other_vec_host[i] = (half_float::half)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  ACL_CHECK_RET(aclrtMemcpyAsync(input_workspace, GetShapeSize(input_shape) * sizeof(half_float::half),
                                 input_vec_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtMemcpyAsync(other_workspace, GetShapeSize(input_shape) * sizeof(half_float::half),
                                 other_vec_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  Mul(input_tensor, other_tensor, &output_tensor, stream, llm_kernels::utils::GetTestWorkSpaceFunc);

  ACL_CHECK_RET(aclrtMemcpyAsync(out_vec_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 output_workspace, GetShapeSize(output_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  for (size_t i = 0; i < input_vec_host.size(); ++i) {
    EXPECT_NEAR(float(input_vec_host[i]) * float(other_vec_host[i]), float(out_vec_host[i]), 1e-3);
  }

  ACL_CHECK_RET(aclDestroyTensor(output_tensor));
  ACL_CHECK_RET(aclDestroyTensor(input_tensor));
  ACL_CHECK_RET(aclDestroyTensor(other_tensor));
  ACL_CHECK_RET(aclrtFree(input_workspace));
  ACL_CHECK_RET(aclrtFree(output_workspace));
  ACL_CHECK_RET(aclrtFree(other_workspace));
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
