/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "aclrtlaunch_InvokeCastFloatToHalfKernel.h"
#include "csrc/kernels/ascend/cast/cast_tiling.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendCastTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::stream;
};

TEST_F(LlamaAscendCastTestSuit, CommonTest) {
  uint32_t total_elem_num = 32 * 5120;
  // element number in each block
  constexpr uint32_t block_elem_num = 1 * 5120;
  uint32_t dim_num = total_elem_num / block_elem_num;
  CastTilingConfig tiling;
  tiling.total_elem_num = total_elem_num;
  tiling.block_elem_num = block_elem_num;
  tiling.tile_num = 2;

  using SRC_DTYPE = float;
  using DST_DTYPE = half_float::half;

  size_t input_size = total_elem_num * sizeof(SRC_DTYPE);
  uint8_t *input_host;
  uint8_t *input_device;
  std::vector<SRC_DTYPE> input_ref(total_elem_num);
  ACL_CHECK_RET(aclrtMallocHost((void **)(&input_host), input_size));
  ACL_CHECK_RET(aclrtMalloc((void **)&input_device, input_size, ACL_MEM_MALLOC_HUGE_FIRST));
  for (size_t i = 0; i < total_elem_num; ++i) {
    ((SRC_DTYPE *)input_host)[i] = SRC_DTYPE(std::sin(float(i)));
    input_ref[i] = ((SRC_DTYPE *)input_host)[i];
  }
  ACL_CHECK_RET(aclrtMemcpy(input_device, input_size, input_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE));

  size_t output_size = total_elem_num * sizeof(DST_DTYPE);
  uint8_t *output_host;
  uint8_t *output_device;
  std::vector<DST_DTYPE> output_ref(total_elem_num);
  ACL_CHECK_RET(aclrtMallocHost((void **)(&output_host), output_size));
  ACL_CHECK_RET(aclrtMalloc((void **)&output_device, output_size, ACL_MEM_MALLOC_HUGE_FIRST));
  for (size_t i = 0; i < total_elem_num; ++i) {
    output_ref[i] = DST_DTYPE(input_ref[i]);
  }

  CastTilingConfig *buf = &tiling;
  size_t tiling_size = sizeof(CastTilingConfig);
  uint8_t *tiling_device;
  ACL_CHECK_RET(aclrtMalloc((void **)&tiling_device, tiling_size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(tiling_device, tiling_size, (void *)buf, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE));

  ACL_CHECK_RET(
      ACLRT_LAUNCH_KERNEL(InvokeCastFloatToHalfKernel)(dim_num, stream, input_device, output_device, tiling_device));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  ACL_CHECK_RET(aclrtMemcpy(output_host, output_size, output_device, output_size, ACL_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0; i < total_elem_num; ++i) {
    EXPECT_NEAR(((DST_DTYPE *)output_host)[i], output_ref[i], 1e-5);
  }

  ACL_CHECK_RET(aclrtFree(tiling_device));
  ACL_CHECK_RET(aclrtFree(output_device));
  ACL_CHECK_RET(aclrtFreeHost(output_host));
  ACL_CHECK_RET(aclrtFree(input_device));
  ACL_CHECK_RET(aclrtFreeHost(input_host));
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels