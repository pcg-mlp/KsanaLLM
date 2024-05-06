/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "aclrtlaunch_InvokeMatmulKernel.h"
#include "csrc/kernels/ascend/matmul/matmul.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"
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

  template <typename T>
  void CalcMatmulRef(T *a_ptr, T *b_ptr, T *c_ptr, size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        T sum = (T)(0.0);
        for (size_t s = 0; s < k; ++s) {
          sum += (a_ptr[i * k + s] * b_ptr[s * n + j]);
        }
        c_ptr[i * n + j] = sum;
      }
    }
  }
};

TEST_F(LlamaAscendMatmulTestSuit, AclNNMatmulTest) {
  size_t m = 1;
  size_t n = 1;
  size_t k = 2;
  const std::vector<int64_t> input_shape = {m, k};
  aclTensor *input_tensor = nullptr;
  void *input_workspace = nullptr;
  const std::vector<int64_t> other_shape = {k, n};
  aclTensor *other_tensor = nullptr;
  void *other_workspace = nullptr;
  const std::vector<int64_t> output_shape = {m, n};
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
  size_t m = 16ul;
  size_t n = 16ul;
  size_t k = 16ul;
  size_t a_size = m * k * sizeof(half_float::half);
  size_t b_size = k * n * sizeof(half_float::half);
  size_t c_size = m * n * sizeof(float);
  uint32_t block_dim = 16;
  uint8_t *a_host;
  uint8_t *a_device;
  std::vector<float> a_ref(m * k);
  ACL_CHECK_RET(aclrtMallocHost((void **)(&a_host), a_size));
  ACL_CHECK_RET(aclrtMalloc((void **)&a_device, a_size, ACL_MEM_MALLOC_HUGE_FIRST));

  for (size_t i = 0; i < m * k; ++i) {
    a_ref[i] = float(std::sin(float(i)));
    ((half_float::half *)a_host)[i] = (half_float::half)(a_ref[i]);
  }
  ACL_CHECK_RET(aclrtMemcpy(a_device, a_size, a_host, a_size, ACL_MEMCPY_HOST_TO_DEVICE));

  uint8_t *b_host;
  uint8_t *b_device;
  std::vector<float> b_ref(k * n);
  ACL_CHECK_RET(aclrtMallocHost((void **)(&b_host), b_size));
  ACL_CHECK_RET(aclrtMalloc((void **)&b_device, b_size, ACL_MEM_MALLOC_HUGE_FIRST));
  for (size_t i = 0; i < m * k; ++i) {
    b_ref[i] = float(std::cos(float(i)));
    ((half_float::half *)b_host)[i] = (half_float::half)(b_ref[i]);
  }
  ACL_CHECK_RET(aclrtMemcpy(b_device, b_size, b_host, b_size, ACL_MEMCPY_HOST_TO_DEVICE));

  matmul_tiling::TPosition left_pos = matmul_tiling::TPosition::GM;
  matmul_tiling::CubeFormat left_format = matmul_tiling::CubeFormat::ND;
  matmul_tiling::DataType left_dtype = matmul_tiling::DataType::DT_FLOAT16;
  bool transpose_a = false;
  matmul_tiling::TPosition right_pos = matmul_tiling::TPosition::GM;
  matmul_tiling::CubeFormat right_format = matmul_tiling::CubeFormat::ND;
  matmul_tiling::DataType right_dtype = matmul_tiling::DataType::DT_FLOAT16;
  bool transpose_b = false;
  matmul_tiling::TPosition res_pos = matmul_tiling::TPosition::GM;
  matmul_tiling::CubeFormat res_format = matmul_tiling::CubeFormat::ND;
  matmul_tiling::DataType res_dtype = matmul_tiling::DataType::DT_FLOAT;
  matmul_tiling::TPosition bias_pos = matmul_tiling::TPosition::GM;
  matmul_tiling::CubeFormat bias_format = matmul_tiling::CubeFormat::ND;
  matmul_tiling::DataType bias_dtype = matmul_tiling::DataType::DT_FLOAT;
  bool is_bias = false;
  int used_core_num = 1;
  optiling::TCubeTiling tiling_data;
  tiling_data.set_usedCoreNum(used_core_num);
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
  EXPECT_GE(buf_size.ubSize, 0);
  EXPECT_GT(buf_size.l0cSize, 0);
  EXPECT_GT(buf_size.l1Size, 0);

  uint32_t tiling_size = tiling_data.GetDataSize();
  char *buf = (char *)malloc(tiling_size);
  tiling_data.SaveToBuffer(buf, tiling_size);

  uint8_t *tiling_device;
  ACL_CHECK_RET(aclrtMalloc((void **)&tiling_device, tiling_size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(tiling_device, tiling_size, (void *)buf, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK_RET(aclrtSynchronizeDevice());

  uint8_t *c_host;
  uint8_t *c_device;
  std::vector<float> c_ref(m * n);
  ACL_CHECK_RET(aclrtMallocHost((void **)(&c_host), c_size));
  ACL_CHECK_RET(aclrtMalloc((void **)&c_device, c_size, ACL_MEM_MALLOC_HUGE_FIRST));

  uint8_t *ref_host;
  ACL_CHECK_RET(aclrtMallocHost((void **)(&ref_host), c_size));

  ACL_CHECK_RET(
      ACLRT_LAUNCH_KERNEL(InvokeMatmulKernel)(block_dim, stream, a_device, b_device, c_device, tiling_device));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  ACL_CHECK_RET(aclrtMemcpy(c_host, c_size, c_device, c_size, ACL_MEMCPY_DEVICE_TO_HOST));
  CalcMatmulRef<float>(a_ref.data(), b_ref.data(), c_ref.data(), m, n, k);

  for (size_t i = 0; i < m * n; ++i) {
    EXPECT_NEAR(c_ref[i], ((float *)c_host)[i], 1e-2);
  }

  ACL_CHECK_RET(aclrtFree(tiling_device));
  ACL_CHECK_RET(aclrtFree(c_device));
  ACL_CHECK_RET(aclrtFreeHost(c_host));
  ACL_CHECK_RET(aclrtFree(b_device));
  ACL_CHECK_RET(aclrtFreeHost(b_host));
  ACL_CHECK_RET(aclrtFree(a_device));
  ACL_CHECK_RET(aclrtFreeHost(a_host));
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
