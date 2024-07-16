/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "csrc/kernels/ascend/matmul/matmul.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"
#include "tests/references/matmul.h"
#include "tiling/tiling_api.h"

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

  template <typename DTYPE>
  void RunAclNNMatmulTest() {
    aclDataType aclnn_dtype = aclDataType::ACL_FLOAT16;
    if (std::is_same<DTYPE, float>::value) {
      aclnn_dtype = aclDataType::ACL_FLOAT;
    } else if (std::is_same<DTYPE, aclFloat16>::value || std::is_same<DTYPE, half_float::half>::value) {
      aclnn_dtype = aclDataType::ACL_FLOAT16;
    } else {
      GTEST_SKIP_("This test is just supported float and float16.");
    }

    size_t m = 128;
    size_t n = 1024;
    size_t k = 1024;
    llm_kernels::utils::ACLNNMatmulComputeType mm_type = llm_kernels::utils::ACLNNMatmulComputeType::KEEP_DTYPE;
    if (std::is_same<DTYPE, aclFloat16>::value || std::is_same<DTYPE, half_float::half>::value) {
      mm_type = llm_kernels::utils::ACLNNMatmulComputeType::USE_FP16;
    }

    const std::vector<int64_t> input_shape = {static_cast<int64_t>(m), static_cast<int64_t>(k)};
    aclTensor *input_tensor = nullptr;
    void *input_workspace = nullptr;
    const std::vector<int64_t> other_shape = {static_cast<int64_t>(k), static_cast<int64_t>(n)};
    aclTensor *other_tensor = nullptr;
    void *other_workspace = nullptr;
    const std::vector<int64_t> output_shape = {static_cast<int64_t>(m), static_cast<int64_t>(n)};
    aclTensor *output_tensor = nullptr;
    void *output_workspace = nullptr;
    void *output_host = nullptr;
    ACL_CHECK_RET(aclrtMallocHost((void **)(&output_host), GetShapeSize(output_shape) * sizeof(DTYPE)));
    CreateAclTensor(input_shape, &input_workspace, aclnn_dtype, aclFormat::ACL_FORMAT_ND, &input_tensor);
    CreateAclTensor(other_shape, &other_workspace, aclnn_dtype, aclFormat::ACL_FORMAT_ND, &other_tensor);
    CreateAclTensor(output_shape, &output_workspace, aclnn_dtype, aclFormat::ACL_FORMAT_ND, &output_tensor);
    std::vector<DTYPE> input_vec_host(GetShapeSize(input_shape));
    std::vector<DTYPE> weight_vec_host(GetShapeSize(other_shape));
    std::vector<DTYPE> output_vec_host(GetShapeSize(output_shape));
    for (size_t i = 0; i < input_vec_host.size(); ++i) {
      if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        input_vec_host[i] = (DTYPE)(std::sin(float(i)));
      } else {
        input_vec_host[i] = aclFloatToFloat16(std::sin(float(i)));
      }
    }
    for (size_t i = 0; i < input_vec_host.size(); ++i) {
      if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        weight_vec_host[i] = (DTYPE)(std::cos(float(i)));
      } else {
        weight_vec_host[i] = aclFloatToFloat16(std::cos(float(i)));
      }
    }
    ACL_CHECK_RET(aclrtMemcpyAsync(input_workspace, GetShapeSize(input_shape) * sizeof(DTYPE), input_vec_host.data(),
                                   GetShapeSize(input_shape) * sizeof(DTYPE), ACL_MEMCPY_HOST_TO_DEVICE, stream));
    ACL_CHECK_RET(aclrtMemcpyAsync(other_workspace, GetShapeSize(other_shape) * sizeof(DTYPE), weight_vec_host.data(),
                                   GetShapeSize(other_shape) * sizeof(DTYPE), ACL_MEMCPY_HOST_TO_DEVICE, stream));
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));

    InvokeAclNNMatMul(input_tensor, other_tensor, mm_type, &output_tensor, stream,
                      llm_kernels::utils::GetTestWorkSpaceFunc);

    ACL_CHECK_RET(aclrtMemcpyAsync(output_host, GetShapeSize(output_shape) * sizeof(DTYPE), output_workspace,
                                   GetShapeSize(output_shape) * sizeof(DTYPE), ACL_MEMCPY_DEVICE_TO_HOST, stream));
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));
    CalcMatmulRef<DTYPE>(input_vec_host.data(), weight_vec_host.data(), output_vec_host.data(), m, n, k);
    for (size_t i = 0; i < m * n; ++i) {
      EXPECT_NEAR(output_vec_host[i], ((DTYPE *)(output_host))[i], 1e-1);
    }

    ACL_CHECK_RET(aclrtFreeHost(output_host));
    ACL_CHECK_RET(aclDestroyTensor(output_tensor));
    ACL_CHECK_RET(aclDestroyTensor(input_tensor));
    ACL_CHECK_RET(aclDestroyTensor(other_tensor));
    ACL_CHECK_RET(aclrtFree(input_workspace));
    ACL_CHECK_RET(aclrtFree(output_workspace));
    ACL_CHECK_RET(aclrtFree(other_workspace));
  }
};

TEST_F(LlamaAscendMatmulTestSuit, AclNNMatmulTest) {
  RunAclNNMatmulTest<half_float::half>();
  RunAclNNMatmulTest<float>();
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
