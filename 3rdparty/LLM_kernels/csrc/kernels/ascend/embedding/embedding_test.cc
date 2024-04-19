/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include <vector>

#include "3rdparty/half.hpp"
#include "csrc/kernels/ascend/embedding/embedding.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

#include "aclnnop/aclnn_copy.h"

namespace llm_kernels {
namespace ascend {
namespace test {

class AscendEmbeddingTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::stream;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& data, const std::vector<int64_t>& shape, void** addr, aclDataType dtype,
                    aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  ACL_CHECK_RET(aclrtMalloc(addr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(*addr, size, data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *addr);
  return 0;
}

TEST_F(AscendEmbeddingTestSuit, EmbeddingLookupTest) {
  // Create weight [100, 16]
  std::vector<float> weights;
  weights.reserve(100 * 16);
  for (size_t i = 0; i < 100; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      weights.push_back(i + (0.01 * j));
    }
  }
  std::vector<int64_t> weight_shape = {100, 16};

  // Create input ids.
  std::vector<int64_t> input_shape = {4};
  std::vector<int64_t> inputs = {8, 48, 68, 88};

  // Create output.
  std::vector<int64_t> out_shape = {4, 16};
  std::vector<float> outs(4 * 16, 0);

  // device addr.
  void* weight_dev_addr = nullptr;
  void* input_dev_addr = nullptr;
  void* out_dev_addr = nullptr;

  // Create tensors.
  aclTensor* weight_tensor = nullptr;
  aclTensor* input_tensor = nullptr;
  aclTensor* out_tensor = nullptr;

  ACL_CHECK_RET(CreateAclTensor(weights, weight_shape, &weight_dev_addr, aclDataType::ACL_FLOAT, &weight_tensor));
  ACL_CHECK_RET(CreateAclTensor(inputs, input_shape, &input_dev_addr, aclDataType::ACL_INT64, &input_tensor));
  ACL_CHECK_RET(CreateAclTensor(outs, out_shape, &out_dev_addr, aclDataType::ACL_FLOAT, &out_tensor));

  LookupEmbedding(input_tensor, weight_tensor, nullptr, out_tensor, stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  // Check result
  std::vector<float> result(4 * 16, 0);
  ACL_CHECK_RET(aclrtMemcpy(result.data(), result.size() * sizeof(float), out_dev_addr, 4 * 16 * sizeof(float),
                            ACL_MEMCPY_DEVICE_TO_HOST));

  size_t idx = 0;
  for (auto i : inputs) {
    for (size_t j = 0; j < 16; ++j) {
      EXPECT_FLOAT_EQ(i + (0.01 * j), result[idx++]);
    }
  }
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
