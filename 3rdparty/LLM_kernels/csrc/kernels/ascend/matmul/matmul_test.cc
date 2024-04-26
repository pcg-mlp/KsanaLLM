/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "csrc/kernels/ascend/matmul/matmul.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendMatmulTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::context;
  using AscendTestSuitBase::default_device;
  using AscendTestSuitBase::is_inited;
  using AscendTestSuitBase::stream;
};

TEST_F(LlamaAscendMatmulTestSuit, MatmulTest) {
  const std::vector<int64_t> input_shape = {1, 2};
  aclTensor* input_tensor = nullptr;
  void* input_workspace = nullptr;
  const std::vector<int64_t> other_shape = {2, 1};
  aclTensor* other_tensor = nullptr;
  void* other_workspace = nullptr;
  const std::vector<int64_t> output_shape = {1, 1};
  aclTensor* output_tensor = nullptr;
  void* output_workspace = nullptr;
  CreateAclTensor(input_shape, &input_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &input_tensor);
  CreateAclTensor(other_shape, &other_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &other_tensor);
  CreateAclTensor(output_shape, &output_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &output_tensor);
  std::vector<half_float::half> input_vec_host(GetShapeSize(input_shape));
  std::vector<half_float::half> other_vec_host(GetShapeSize(other_shape));
  std::vector<half_float::half> output_vec_host(GetShapeSize(output_shape));
  for (size_t i = 0; i < input_vec_host.size(); ++i) {
    input_vec_host[i] = (half_float::half)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    other_vec_host[i] = (half_float::half)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  ACL_CHECK_RET(aclrtMemcpyAsync(input_workspace, GetShapeSize(input_shape) * sizeof(half_float::half),
                                 input_vec_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtMemcpyAsync(other_workspace, GetShapeSize(other_shape) * sizeof(half_float::half),
                                 other_vec_host.data(), GetShapeSize(other_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  int mm_type = 0;
  MatMul(input_tensor, other_tensor, mm_type, &output_tensor, stream, llm_kernels::utils::GetTestWorkSpaceFunc);

  ACL_CHECK_RET(aclrtMemcpyAsync(output_vec_host.data(), GetShapeSize(output_shape) * sizeof(half_float::half),
                                 output_workspace, GetShapeSize(output_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  for (size_t m = 0; m < input_shape[0]; ++m) {
    for (size_t n = 0; n < other_shape[1]; ++n) {
      float sum = 0.0f;
      for (size_t k = 0; k < input_shape[1]; ++k) {
        sum += (input_vec_host[m * input_shape[1] + k] * other_vec_host[k * other_shape[1] + n]);
      }
      EXPECT_NEAR(sum, float(output_vec_host[m * input_shape[0] + n]), 1e-3);
    }
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
