/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "csrc/kernels/ascend/slice/slice.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendSliceTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::context;
  using AscendTestSuitBase::default_device;
  using AscendTestSuitBase::is_inited;
  using AscendTestSuitBase::stream;
};

TEST_F(LlamaAscendSliceTestSuit, SliceTest) {
  const std::vector<int64_t> input_shape = {4, 8};
  aclTensor* input_tensor = nullptr;
  void* input_workspace = nullptr;

  const std::vector<int64_t> output_shape = {4, 4};
  aclTensor* output_tensor = nullptr;
  void* output_workspace = nullptr;

  CreateAclTensor(input_shape, &input_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &input_tensor);
  CreateAclTensor(output_shape, &output_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &output_tensor);
  std::vector<half_float::half> input_vec_host(GetShapeSize(input_shape));
  std::vector<half_float::half> out_vec_host(GetShapeSize(output_shape));
  for (size_t i = 0; i < input_vec_host.size(); ++i) {
    input_vec_host[i] = (half_float::half)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  ACL_CHECK_RET(aclrtMemcpyAsync(input_workspace, GetShapeSize(input_shape) * sizeof(half_float::half),
                                 input_vec_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  Slice(input_tensor, -1, 4, 8, 1, &output_tensor, stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  ACL_CHECK_RET(aclrtMemcpyAsync(out_vec_host.data(), GetShapeSize(output_shape) * sizeof(half_float::half),
                                 output_workspace, GetShapeSize(output_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  for (size_t i = 0; i < input_shape[0]; ++i) {
    for (size_t j = 4; j < input_shape[1]; ++j) {
      EXPECT_NEAR(float(input_vec_host[i * input_shape[1] + j]), float(out_vec_host[i * output_shape[1] + j - 4]),
                  1e-5);
    }
  }

  ACL_CHECK_RET(aclDestroyTensor(output_tensor));
  ACL_CHECK_RET(aclDestroyTensor(input_tensor));
  ACL_CHECK_RET(aclrtFree(input_workspace));
  ACL_CHECK_RET(aclrtFree(output_workspace));
}

TEST_F(LlamaAscendSliceTestSuit, SliceKernelTest) {
  // A [10, 10] matrix.
  int row = 10;
  int col = 10;

  std::vector<float> data(row * col, 0.0);
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      data[i * row + j] = i + (0.001 * j);
    }
  }

  void* input_data_dev;
  size_t input_size = row * col * sizeof(float);
  ACL_CHECK_RET(aclrtMalloc(&input_data_dev, input_size + 32, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(input_data_dev, input_size, data.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE));

  // A [10, 3] matrix
  int output_row = 10;
  int output_col = 3;

  void* output_data_dev;
  size_t output_size = output_row * output_col * sizeof(float);
  ACL_CHECK_RET(aclrtMalloc(&output_data_dev, output_size + 32, ACL_MEM_MALLOC_HUGE_FIRST));

  Slice2 slice;
  uint32_t start_offset = 3 * sizeof(float);
  uint32_t slice_length = 3 * sizeof(float);
  uint32_t slice_step = 10 * sizeof(float);
  uint32_t slice_times = 10;
  slice.Forward(output_data_dev, input_data_dev, start_offset, slice_length, slice_step, slice_times, stream);
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  std::vector<float> result(output_row * output_col, 0);
  ACL_CHECK_RET(aclrtMemcpy(result.data(), result.size() * sizeof(float), output_data_dev,
                            output_row * output_col * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

  size_t idx = 0;
  for (int i = 0; i < output_row; ++i) {
    for (int j = 0; j < output_col; ++j) {
      EXPECT_FLOAT_EQ(i + (0.001 * (3 + j)), result[idx++]);
    }
  }
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
