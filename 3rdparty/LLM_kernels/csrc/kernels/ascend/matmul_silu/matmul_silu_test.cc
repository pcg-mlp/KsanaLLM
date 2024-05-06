/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <cmath>
#include <gtest/gtest.h>

#include "3rdparty/half/include/half.hpp"
#include "aclrtlaunch_InvokeMatmulSiluKernel.h"
#include "csrc/kernels/ascend/matmul_silu/matmul_silu.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendMatmulSiluTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::stream;
};

TEST_F(LlamaAscendMatmulSiluTestSuit, CommonTest) {
  constexpr uint32_t block_dim = 8;
  constexpr uint32_t row_dim = 1024;
  constexpr uint32_t col_dim = 1024;
  size_t input_byte_size = row_dim * col_dim * sizeof(aclFloat16);
  size_t output_byte_size = row_dim * col_dim * sizeof(aclFloat16);
  aclFloat16* x_host;
  aclFloat16* y_host;
  aclFloat16* z_host;
  ACL_CHECK_RET(aclrtMallocHost((void**)(&x_host), input_byte_size));
  ACL_CHECK_RET(aclrtMallocHost((void**)(&y_host), input_byte_size));
  ACL_CHECK_RET(aclrtMallocHost((void**)(&z_host), output_byte_size));
  for (uint32_t i = 0; i < row_dim * col_dim; ++i) {
      x_host[i] = aclFloatToFloat16(float(std::cos(double(i))));
      y_host[i] = aclFloatToFloat16(float(std::sin(double(i))));
  }
  aclFloat16* x_device;
  aclFloat16* y_device;
  aclFloat16* z_device;
  aclFloat16* bias_device;
  aclFloat16* workspace_device;
  aclFloat16* tiling_device;
  ACL_CHECK_RET(aclrtMalloc((void**)&x_device, input_byte_size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMalloc((void**)&y_device, input_byte_size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMalloc((void**)&z_device, output_byte_size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMalloc((void**)&workspace_device, output_byte_size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMalloc((void**)&tiling_device, output_byte_size, ACL_MEM_MALLOC_HUGE_FIRST));

  ACL_CHECK_RET(aclrtMemcpy(x_device, input_byte_size, x_host, input_byte_size, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK_RET(aclrtMemcpy(y_device, input_byte_size, y_host, input_byte_size, ACL_MEMCPY_HOST_TO_DEVICE));

  // TODO(karlluo): implement
  // ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeMatmulSiluKernel)(block_dim, stream, x_device, y_device, bias_device,
  //                                                           z_device, workspace_device, tiling_device));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  ACL_CHECK_RET(aclrtFree(x_device));
  ACL_CHECK_RET(aclrtFree(y_device));
  ACL_CHECK_RET(aclrtFree(z_device));
  ACL_CHECK_RET(aclrtFreeHost(x_host));
  ACL_CHECK_RET(aclrtFreeHost(y_host));
  ACL_CHECK_RET(aclrtFreeHost(z_host));
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
