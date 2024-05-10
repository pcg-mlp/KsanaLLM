/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "aclrtlaunch_InvokeRmsNormKernel.h"
#include "csrc/kernels/ascend/rmsnorm/rmsnorm.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendRmsNormTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::stream;
};

template <typename T>
T ClampInfForHalf(const float input) {
  return input;
}

template <>
half_float::half ClampInfForHalf(const float input) {
  // clamp inf values to enable fp16 training
  return input > 0.0f ? (half_float::half)std::min(input, HALF_FLT_MAX - 1000)
                      : (half_float::half)std::max(input, -HALF_FLT_MAX + 1000);
}

template <typename T>
void RmsNormRef(const T *input, const T *gamma, const float eps, const size_t m, const size_t n, T *output) {
  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    float var_sum = 0.0f;
    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
      var_sum += float(input[m_idx * n + n_idx] * input[m_idx * n + n_idx]);
    }
    float s_variance = 1.0f / std::sqrt(var_sum / (float)n + eps);
    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
      output[m_idx * n + n_idx] = ClampInfForHalf<T>((input[m_idx * n + n_idx] * s_variance) * (float)(gamma[n_idx]));
    }
  }
}

TEST_F(LlamaAscendRmsNormTestSuit, RmsNormKernelTest) {
  size_t m = 2;
  size_t n = 1024;

  using dtype = half_float::half;

  size_t input_size = m * n * sizeof(dtype);
  uint8_t *input_host;
  uint8_t *input_device;
  std::vector<dtype> input_ref(m * n);
  ACL_CHECK_RET(aclrtMallocHost((void **)(&input_host), input_size));
  ACL_CHECK_RET(aclrtMalloc((void **)&input_device, input_size, ACL_MEM_MALLOC_HUGE_FIRST));
  for (size_t i = 0; i < m * n; ++i) {
    ((dtype *)input_host)[i] = dtype(std::sin(float(i)));
    input_ref[i] = ((dtype *)input_host)[i];
  }
  ACL_CHECK_RET(aclrtMemcpy(input_device, input_size, input_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE));

  size_t gamma_size = n * sizeof(dtype);
  uint8_t *gamma_host;
  uint8_t *gamma_device;
  std::vector<dtype> gamma_ref(n);
  ACL_CHECK_RET(aclrtMallocHost((void **)(&gamma_host), gamma_size));
  ACL_CHECK_RET(aclrtMalloc((void **)&gamma_device, gamma_size, ACL_MEM_MALLOC_HUGE_FIRST));
  for (size_t i = 0; i < n; ++i) {
    ((dtype *)gamma_host)[i] = dtype(std::cos(float(i)));
    gamma_ref[i] = ((dtype *)gamma_host)[i];
  }
  ACL_CHECK_RET(aclrtMemcpy(gamma_device, gamma_size, gamma_host, gamma_size, ACL_MEMCPY_HOST_TO_DEVICE));

  size_t output_size = m * n * sizeof(dtype);
  uint8_t *output_host;
  uint8_t *output_device;
  std::vector<dtype> output_ref(m * n);
  ACL_CHECK_RET(aclrtMallocHost((void **)(&output_host), output_size));
  ACL_CHECK_RET(aclrtMalloc((void **)&output_device, output_size, ACL_MEM_MALLOC_HUGE_FIRST));

  // calc ref
  RmsNormRef<half_float::half>(input_ref.data(), gamma_ref.data(), 1e-6, m, n, output_ref.data());

  // NOTE(karlluo): m is seq length, n is hidden units number
  RmsNormTilingConfig tiling;
  tiling.bLength = 1;
  tiling.sLength = m;
  tiling.hLength = n;
  tiling.originalHLength = n;
  tiling.loopRound = 1;
  tiling.eps = 1e-6;
  // NOTE(karlluo): relate to reduce sum buffer size
  tiling.mainBsLengthAlign = n * sizeof(float);
  // NOTE(karlluo): relate to x² fp32 buffer size
  tiling.mainBshLength = n * sizeof(float);
  tiling.mainBsLength = 1;
  tiling.reciprocalOfHLength = float(1.0f) / float(n);
  RmsNormTilingConfig *buf = &tiling;
  size_t tiling_size = sizeof(RmsNormTilingConfig);
  uint8_t *tiling_device;
  ACL_CHECK_RET(aclrtMalloc((void **)&tiling_device, tiling_size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(tiling_device, tiling_size, (void *)buf, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE));

  // reduce space
  uint32_t workspace_buffer_len = tiling.mainBsLengthAlign;
  if constexpr (sizeof(dtype) == sizeof(half_float::half)) {
    // tmp space
    workspace_buffer_len += tiling.mainBshLength;
  }
  // fp32 tmp space
  workspace_buffer_len += tiling.hLength;
  workspace_buffer_len *= m;
  uint8_t *workspace_device;
  ACL_CHECK_RET(aclrtMalloc((void **)&workspace_device, workspace_buffer_len, ACL_MEM_MALLOC_HUGE_FIRST));

  ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeRmsNormKernel)(m, stream, input_device, gamma_device, output_device,
                                                         tiling_device, workspace_device));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  ACL_CHECK_RET(aclrtMemcpy(output_host, output_size, output_device, output_size, ACL_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0; i < m; ++i) {
    dtype *output_ptr = ((dtype *)output_host) + (i * n);
    for (size_t j = 0; j < n; ++j) {
      EXPECT_NEAR(output_ptr[j], output_ref[i * n + j], 1e-3);
    }
  }

  ACL_CHECK_RET(aclrtFree(workspace_device));
  ACL_CHECK_RET(aclrtFree(tiling_device));
  ACL_CHECK_RET(aclrtFree(output_device));
  ACL_CHECK_RET(aclrtFreeHost(output_host));
  ACL_CHECK_RET(aclrtFree(gamma_device));
  ACL_CHECK_RET(aclrtFreeHost(gamma_host));
  ACL_CHECK_RET(aclrtFree(input_device));
  ACL_CHECK_RET(aclrtFreeHost(input_host));
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels