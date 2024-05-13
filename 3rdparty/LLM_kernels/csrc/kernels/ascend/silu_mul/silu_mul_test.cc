/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "aclrtlaunch_InvokeSiluMulHalfKernel.h"
#include "csrc/kernels/ascend/silu_mul/silu_mul_kernel.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendSiluMulTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::stream;

 protected:
  template <typename DTYPE>
  void TestSiluMul() {
    uint32_t total_elem_num = 32 * 5120;
    // element number in each block
    constexpr uint32_t block_elem_num = 1 * 5120;
    uint32_t dim_num = total_elem_num / block_elem_num;
    SiluMulTilingConfig tiling;
    tiling.total_elem_num = total_elem_num;
    tiling.block_elem_num = block_elem_num;
    tiling.tile_num = 2;

    size_t input_size = total_elem_num * sizeof(DTYPE);
    uint8_t *input_host;
    uint8_t *input_device;
    std::vector<DTYPE> input_ref(total_elem_num);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&input_host), input_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&input_device, input_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < total_elem_num; ++i) {
      ((DTYPE *)input_host)[i] = DTYPE(std::sin(float(i)));
      input_ref[i] = ((DTYPE *)input_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(input_device, input_size, input_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t weight_size = total_elem_num * sizeof(DTYPE);
    uint8_t *weight_host;
    uint8_t *weight_device;
    std::vector<DTYPE> weight_ref(total_elem_num);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&weight_host), weight_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&weight_device, weight_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < total_elem_num; ++i) {
      ((DTYPE *)weight_host)[i] = DTYPE(std::cos(float(i)));
      weight_ref[i] = ((DTYPE *)weight_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(weight_device, weight_size, weight_host, weight_size, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t output_size = total_elem_num * sizeof(DTYPE);
    uint8_t *output_host;
    uint8_t *output_device;
    std::vector<DTYPE> output_ref(total_elem_num);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&output_host), output_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&output_device, output_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < total_elem_num; ++i) {
      output_ref[i] =
          DTYPE(float(DTYPE(input_ref[i])) * (1.0f / (1 + std::exp(-float(DTYPE(input_ref[i])))))) * weight_ref[i];
    }

    SiluMulTilingConfig *buf = &tiling;
    size_t tiling_size = sizeof(SiluMulTilingConfig);
    uint8_t *tiling_device;
    ACL_CHECK_RET(aclrtMalloc((void **)&tiling_device, tiling_size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK_RET(aclrtMemcpy(tiling_device, tiling_size, (void *)buf, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeSiluMulHalfKernel)(dim_num, stream, input_device, weight_device,
                                                               output_device, tiling_device));
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));
    ACL_CHECK_RET(aclrtMemcpy(output_host, output_size, output_device, output_size, ACL_MEMCPY_DEVICE_TO_HOST));

    for (size_t i = 0; i < total_elem_num; ++i) {
      EXPECT_NEAR(((DTYPE *)output_host)[i], output_ref[i], 1e-3);
    }

    ACL_CHECK_RET(aclrtFree(weight_device));
    ACL_CHECK_RET(aclrtFreeHost(weight_host));
    ACL_CHECK_RET(aclrtFree(tiling_device));
    ACL_CHECK_RET(aclrtFree(output_device));
    ACL_CHECK_RET(aclrtFreeHost(output_host));
    ACL_CHECK_RET(aclrtFree(input_device));
    ACL_CHECK_RET(aclrtFreeHost(input_host));
  }
};

TEST_F(LlamaAscendSiluMulTestSuit, CommonTest) { TestSiluMul<half_float::half>(); }

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels