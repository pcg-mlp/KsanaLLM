/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "aclrtlaunch_InvokeMatmulFloatKernel.h"
#include "aclrtlaunch_InvokeMatmulHalfKernel.h"
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
  void RunMatMulTest() {
    size_t m = 32ul;
    size_t n = 5120ul;
    size_t k = 1024ul;
    size_t a_size = m * k * sizeof(DTYPE);
    size_t b_size = k * n * sizeof(DTYPE);
    size_t c_size = m * n * sizeof(DTYPE);
    uint32_t block_dim = CUBE_CORE_NUM;

    uint8_t *a_host;
    uint8_t *a_device;
    std::vector<DTYPE> a_ref(m * k);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&a_host), a_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&a_device, a_size, ACL_MEM_MALLOC_HUGE_FIRST));

    for (size_t i = 0; i < m * k; ++i) {
      if (std::is_same<DTYPE, half_float::half>::value || std::is_same<DTYPE, float>::value) {
        a_ref[i] = DTYPE(std::sin(float(i)));
        ((DTYPE *)a_host)[i] = (DTYPE)(a_ref[i]);
      } else if (std::is_same<DTYPE, aclFloat16>::value) {
        a_ref[i] = aclFloatToFloat16(std::sin(float(i)));
        ((DTYPE *)a_host)[i] = a_ref[i];
      } else {
        throw std::invalid_argument("Not support matmul dtype, only support float16 and float32");
      }
    }
    ACL_CHECK_RET(aclrtMemcpy(a_device, a_size, a_host, a_size, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *b_host;
    uint8_t *b_device;
    std::vector<DTYPE> b_ref(k * n);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&b_host), b_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&b_device, b_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < k * n; ++i) {
      if (std::is_same<DTYPE, half_float::half>::value || std::is_same<DTYPE, float>::value) {
        b_ref[i] = DTYPE(std::cos(float(i)));
        ((DTYPE *)b_host)[i] = (DTYPE)(b_ref[i]);
      } else if (std::is_same<DTYPE, aclFloat16>::value) {
        b_ref[i] = aclFloatToFloat16(std::cos(float(i)));
        ((DTYPE *)b_host)[i] = b_ref[i];
      } else {
        throw std::invalid_argument("Not support matmul dtype, only support float16 and float32");
      }
    }
    ACL_CHECK_RET(aclrtMemcpy(b_device, b_size, b_host, b_size, ACL_MEMCPY_HOST_TO_DEVICE));

    matmul_tiling::DataType computing_dtype = matmul_tiling::DataType::DT_FLOAT16;
    if (std::is_same<DTYPE, half_float::half>::value || std::is_same<DTYPE, aclFloat16>::value) {
      computing_dtype = matmul_tiling::DataType::DT_FLOAT16;
    } else if (std::is_same<DTYPE, float>::value) {
      computing_dtype = matmul_tiling::DataType::DT_FLOAT;
    } else {
      throw std::invalid_argument("Not support matmul dtype, only support float16 and float32");
    }

    matmul_tiling::TPosition left_pos = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat left_format = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType left_dtype = computing_dtype;
    bool transpose_a = false;
    matmul_tiling::TPosition right_pos = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat right_format = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType right_dtype = computing_dtype;
    bool transpose_b = false;
    matmul_tiling::TPosition res_pos = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat res_format = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType res_dtype = computing_dtype;
    bool is_bias = false;
    matmul_tiling::TPosition bias_pos = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat bias_format = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType bias_dtype = computing_dtype;

    optiling::TCubeTiling tiling_data;
    tiling_data.set_usedCoreNum(block_dim);
    matmul_tiling::MatmulApiTiling tiling_api;
    tiling_api.SetAType(left_pos, left_format, left_dtype, transpose_a);
    tiling_api.SetBType(right_pos, right_format, right_dtype, transpose_b);
    tiling_api.SetCType(res_pos, res_format, res_dtype);
    tiling_api.SetBiasType(bias_pos, bias_format, bias_dtype);
    tiling_api.SetOrgShape(m, n, k);
    tiling_api.SetShape(m, n, k);
    tiling_api.SetBias(is_bias);
    tiling_api.SetBufferSpace(-1, -1, -1);
    EXPECT_NE(tiling_api.GetTiling(tiling_data), -1);

    matmul_tiling::SysTilingTempBufSize buf_size;
    EXPECT_NE(MultiCoreMatmulGetTmpBufSize(tiling_data, buf_size), -1);

    uint32_t tiling_size = tiling_data.GetDataSize();
    char *buf = (char *)malloc(tiling_size);
    tiling_data.SaveToBuffer(buf, tiling_size);

    uint8_t *tiling_device;
    ACL_CHECK_RET(aclrtMalloc((void **)&tiling_device, tiling_size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK_RET(aclrtMemcpy(tiling_device, tiling_size, (void *)buf, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *c_host;
    uint8_t *c_device;
    std::vector<DTYPE> c_ref(m * n);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&c_host), c_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&c_device, c_size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK_RET(aclrtSynchronizeDevice());

    if (std::is_same<DTYPE, half_float::half>::value || std::is_same<DTYPE, aclFloat16>::value) {
      ACL_CHECK_RET(
          ACLRT_LAUNCH_KERNEL(InvokeMatmulHalfKernel)(block_dim, stream, a_device, b_device, c_device, tiling_device));
    } else if (std::is_same<DTYPE, float>::value) {
      ACL_CHECK_RET(
          ACLRT_LAUNCH_KERNEL(InvokeMatmulFloatKernel)(block_dim, stream, a_device, b_device, c_device, tiling_device));
    } else {
      throw std::invalid_argument("Not support matmul dtype, only support float16 and float32");
    }
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));

    ACL_CHECK_RET(aclrtMemcpy(c_host, c_size, c_device, c_size, ACL_MEMCPY_DEVICE_TO_HOST));
    CalcMatmulRef<DTYPE>(a_ref.data(), b_ref.data(), c_ref.data(), m, n, k);
    ACL_CHECK_RET(aclrtSynchronizeDevice());

    for (size_t i = 0; i < m * n; ++i) {
      float c_ref_val = -1.0f;
      float c_host_val = 1.0f;

      if (std::is_same<DTYPE, half_float::half>::value) {
        c_ref_val = float(c_ref[i]);
        c_host_val = float(((DTYPE *)c_host)[i]);
      } else if (std::is_same<DTYPE, aclFloat16>::value) {
        c_ref_val = aclFloat16ToFloat(c_ref[i]);
        c_host_val = aclFloat16ToFloat(((DTYPE *)c_host)[i]);
      } else if (std::is_same<DTYPE, float>::value) {
        c_ref_val = c_ref[i];
        c_host_val = ((DTYPE *)c_host)[i];
      } else {
        throw std::invalid_argument("Not support matmul dtype, only support float16 and float32");
      }
      EXPECT_NEAR(c_ref_val, c_host_val, 1e-1);
    }

    ACL_CHECK_RET(aclrtFree(tiling_device));
    ACL_CHECK_RET(aclrtFree(c_device));
    ACL_CHECK_RET(aclrtFreeHost(c_host));
    ACL_CHECK_RET(aclrtFree(b_device));
    ACL_CHECK_RET(aclrtFreeHost(b_host));
    ACL_CHECK_RET(aclrtFree(a_device));
    ACL_CHECK_RET(aclrtFreeHost(a_host));
  }
};

TEST_F(LlamaAscendMatmulTestSuit, AclNNMatmulTest) {
  size_t m = 1;
  size_t n = 1;
  size_t k = 2;
  const std::vector<int64_t> input_shape = {static_cast<int64_t>(m), static_cast<int64_t>(k)};
  aclTensor *input_tensor = nullptr;
  void *input_workspace = nullptr;
  const std::vector<int64_t> other_shape = {static_cast<int64_t>(k), static_cast<int64_t>(n)};
  aclTensor *other_tensor = nullptr;
  void *other_workspace = nullptr;
  const std::vector<int64_t> output_shape = {static_cast<int64_t>(m), static_cast<int64_t>(n)};
  aclTensor *output_tensor = nullptr;
  void *output_workspace = nullptr;
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

TEST_F(LlamaAscendMatmulTestSuit, AscendCMatmulTest) {
  RunMatMulTest<half_float::half>();
  RunMatMulTest<float>();
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
