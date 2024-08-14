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

#ifdef ENABLE_ACL_ATB
#  include "csrc/utils/ascend/atb_executor.h"
#endif

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

#ifdef ENABLE_ACL_ATB
  template <typename DTYPE>
  void RunATBLinearTest() {
    aclDataType aclnn_dtype = aclDataType::ACL_FLOAT16;
    if (std::is_same<DTYPE, float>::value) {
      aclnn_dtype = aclDataType::ACL_FLOAT;
    } else if (std::is_same<DTYPE, aclFloat16>::value || std::is_same<DTYPE, half_float::half>::value) {
      aclnn_dtype = aclDataType::ACL_FLOAT16;
    } else {
      GTEST_SKIP_("This test is just supported float and float16.");
    }

    atb::infer::LinearParam linear_param;
    linear_param.transposeA = false;
    linear_param.transposeB = false;
    linear_param.hasBias = false;
    linear_param.outDataType = ACL_DT_UNDEFINED;
    llm_kernels::utils::ATBOperationExecutor atb_op_executor;
    atb_op_executor.Init(default_device, linear_param);
    atb_op_executor.ResetVariantPack();

    size_t m = 128;
    size_t n = 1024;
    size_t k = 1024;

    const std::vector<size_t> input_shape = {static_cast<size_t>(m), static_cast<size_t>(k)};
    void *input_workspace = nullptr;
    ACL_CHECK_RET(aclrtMalloc(&input_workspace, m * k * sizeof(DTYPE), ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<DTYPE> input_host_vec(m * k);
    const std::vector<size_t> weight_shape = {static_cast<size_t>(k), static_cast<size_t>(n)};
    void *weight_workspace = nullptr;
    ACL_CHECK_RET(aclrtMalloc(&weight_workspace, k * n * sizeof(DTYPE), ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<DTYPE> weight_host_vec(k * n);
    const std::vector<size_t> output_shape = {static_cast<size_t>(m), static_cast<size_t>(n)};
    void *output_workspace = nullptr;
    void *output_host_workspace = nullptr;
    ACL_CHECK_RET(aclrtMalloc(&output_workspace, m * n * sizeof(DTYPE), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK_RET(aclrtMallocHost((void **)(&output_host_workspace), m * n * sizeof(DTYPE)));
    std::vector<DTYPE> output_host_vec(m * n);
    for (size_t i = 0; i < m * k; ++i) {
      if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        input_host_vec[i] = DTYPE(std::sin(float(i)));
      } else {
        input_host_vec[i] = aclFloatToFloat16(std::sin(float(i)));
      }
    }
    for (size_t i = 0; i < k * n; ++i) {
      if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        weight_host_vec[i] = DTYPE(std::cos(float(i)));
      } else {
        weight_host_vec[i] = aclFloatToFloat16(std::cos(float(i)));
      }
    }
    ACL_CHECK_RET(aclrtMemcpy(input_workspace, m * k * sizeof(DTYPE), input_host_vec.data(), m * k * sizeof(DTYPE),
                              ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK_RET(aclrtMemcpy(weight_workspace, k * n * sizeof(DTYPE), weight_host_vec.data(), k * n * sizeof(DTYPE),
                              ACL_MEMCPY_HOST_TO_DEVICE));
    CalcMatmulRef<DTYPE>(input_host_vec.data(), weight_host_vec.data(), output_host_vec.data(), m, n, k);

    atb_op_executor.SetInputTensor(input_workspace, input_shape, aclnn_dtype);
    atb_op_executor.SetInputTensor(weight_workspace, weight_shape, aclnn_dtype);
    atb_op_executor.SetOutputTensor(output_workspace, output_shape, aclnn_dtype);
    atb_op_executor.Run(atb_context, llm_kernels::utils::GetTestWorkSpaceFunc);
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));
    ACL_CHECK_RET(aclrtMemcpy(output_host_workspace, m * n * sizeof(DTYPE), output_workspace, m * n * sizeof(DTYPE),
                              ACL_MEMCPY_DEVICE_TO_HOST));
    for (size_t i = 0; i < m * n; ++i) {
      EXPECT_NEAR(output_host_vec[i], reinterpret_cast<DTYPE *>(output_host_workspace)[i], 1e-1);
    }

    ACL_CHECK_RET(aclrtFree(output_workspace));
    ACL_CHECK_RET(aclrtFree(weight_workspace));
    ACL_CHECK_RET(aclrtFree(input_workspace));
  }
#endif
};

TEST_F(LlamaAscendMatmulTestSuit, AclNNMatmulTest) {
  RunAclNNMatmulTest<half_float::half>();
  RunAclNNMatmulTest<float>();
}

#ifdef ENABLE_ACL_ATB
TEST_F(LlamaAscendMatmulTestSuit, ATBMatmulTest) { RunATBLinearTest<half_float::half>(); }
#endif

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
