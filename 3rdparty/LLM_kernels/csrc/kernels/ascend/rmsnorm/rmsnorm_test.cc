/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "aclrtlaunch_InvokeRmsNormFloatKernel.h"
#include "aclrtlaunch_InvokeRmsNormHalfKernel.h"
#include "csrc/kernels/ascend/rmsnorm/rmsnorm_tiling.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"
#include "tests/references/rms_layernorm.h"

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

  template <typename DTYPE>
  void RunRmsNormTest() {
    // size_t m = 2;
    // size_t n = 5120;
    size_t m = 4;
    size_t n = 4096;
    float eps = 1e-6;

    // NOTE(karlluo): m is seq length, n is hidden units number
    RmsNormTilingConfig tiling;
    // for continue batching, we dont need batch size
    tiling.bLength = 1;
    tiling.sLength = m;
    tiling.hLength = n;
    tiling.originalHLength = n;
    tiling.reciprocalOfHLength = float(1.0f) / float(n);
    // TODO(karlluo): how many elements for each reduce sum handle, if n is too large to load to UB, we need split it to
    // multiple loop round.
    tiling.loopRound = 1;
    // NOTE(karlluo): relate to x² fp32 buffer size
    tiling.mainBshLength = n;
    // NOTE(karlluo): relate to reduce sum buffer size
    tiling.mainBsLengthAlign = n;
    tiling.eps = eps;
    tiling.mainBsLength = 1;

    size_t input_size = m * n * sizeof(DTYPE);
    uint8_t *input_host;
    uint8_t *input_device;
    std::vector<DTYPE> input_ref(m * n);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&input_host), input_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&input_device, input_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < m * n; ++i) {
      ((DTYPE *)input_host)[i] = DTYPE(std::sin(float(i)));
      input_ref[i] = ((DTYPE *)input_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(input_device, input_size, input_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t gamma_size = n * sizeof(DTYPE);
    uint8_t *gamma_host;
    uint8_t *gamma_device;
    std::vector<DTYPE> gamma_ref(n);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&gamma_host), gamma_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&gamma_device, gamma_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < n; ++i) {
      ((DTYPE *)gamma_host)[i] = DTYPE(std::cos(float(i)));
      gamma_ref[i] = ((DTYPE *)gamma_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(gamma_device, gamma_size, gamma_host, gamma_size, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t output_size = m * n * sizeof(DTYPE);
    uint8_t *output_host;
    uint8_t *output_device;
    std::vector<DTYPE> output_ref(m * n);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&output_host), output_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&output_device, output_size, ACL_MEM_MALLOC_HUGE_FIRST));

    RmsNormTilingConfig *buf = &tiling;
    size_t tiling_size = sizeof(RmsNormTilingConfig);
    uint8_t *tiling_device;
    ACL_CHECK_RET(aclrtMalloc((void **)&tiling_device, tiling_size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK_RET(aclrtMemcpy(tiling_device, tiling_size, (void *)buf, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK_RET(aclrtSynchronizeDevice());

    if (std::is_same<DTYPE, half_float::half>::value || std::is_same<DTYPE, aclFloat16>::value) {
      ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeRmsNormHalfKernel)(m, stream, input_device, gamma_device, output_device,
                                                                 tiling_device));
    } else if (std::is_same<DTYPE, float>::value) {
      ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeRmsNormFloatKernel)(m, stream, input_device, gamma_device, output_device,
                                                                  tiling_device));
    } else {
      throw std::invalid_argument("Invalid rms norm compute type, only support float16 or float32.");
    }
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));

    ACL_CHECK_RET(aclrtMemcpy(output_host, output_size, output_device, output_size, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK_RET(aclrtSynchronizeDevice());

    // calc ref
    RmsNormRef<DTYPE>(input_ref.data(), gamma_ref.data(), eps, m, n, output_ref.data());
    for (size_t i = 0; i < m; ++i) {
      DTYPE *output_ptr = ((DTYPE *)output_host) + (i * n);
      for (size_t j = 0; j < n; ++j) {
        EXPECT_NEAR(output_ptr[j], output_ref[i * n + j], 1e-3);
      }
    }

    ACL_CHECK_RET(aclrtFree(tiling_device));
    ACL_CHECK_RET(aclrtFree(output_device));
    ACL_CHECK_RET(aclrtFreeHost(output_host));
    ACL_CHECK_RET(aclrtFree(gamma_device));
    ACL_CHECK_RET(aclrtFreeHost(gamma_host));
    ACL_CHECK_RET(aclrtFree(input_device));
    ACL_CHECK_RET(aclrtFreeHost(input_host));
  }
};

TEST_F(LlamaAscendRmsNormTestSuit, RmsNormKernelTest) {
  RunRmsNormTest<half_float::half>();
  RunRmsNormTest<float>();
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels