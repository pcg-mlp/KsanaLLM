/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "aclrtlaunch_InvokeAddFloatKernel.h"
#include "aclrtlaunch_InvokeAddHalfKernel.h"
#include "csrc/kernels/ascend/add/add.h"
#include "csrc/kernels/ascend/add/add_tiling.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendAddTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::stream;

  template <typename DTYPE>
  void RunAddTest() {
    constexpr uint32_t seq_len = 1;
    constexpr uint32_t hidden_units_num = 4096;
    uint32_t total_elem_num = seq_len * hidden_units_num;
    std::vector<size_t> run_shape = {seq_len, hidden_units_num};
    // element number in each block
    constexpr uint32_t block_elem_num = hidden_units_num;
    float alpha = 1.0f;
    DTYPE alpha_value;
    uint32_t dim_num = total_elem_num / block_elem_num;
    AddTilingConfig tiling;
    tiling.total_elem_num = total_elem_num;
    tiling.block_elem_num = block_elem_num;
    if (std::is_same<DTYPE, float>::value) {
      tiling.alpha = alpha;
      alpha_value = alpha;
    } else if (std::is_same<DTYPE, aclFloat16>::value) {
      aclFloat16 *alpha_buf = (aclFloat16 *)(&tiling.alpha);
      alpha_buf[0] = aclFloatToFloat16(alpha);
      alpha_value = aclFloatToFloat16(alpha);
    } else if (std::is_same<DTYPE, half_float::half>::value) {
      half_float::half *alpha_buf = (half_float::half *)(&tiling.alpha);
      alpha_buf[0] = half_float::half(alpha);
      alpha_value = half_float::half(alpha);
    } else {
      throw std::invalid_argument("Invalid add type type, only support float16 or float32.");
    }

    size_t input_size = total_elem_num * sizeof(DTYPE);

    uint8_t *input_a_host;
    uint8_t *input_a_device;
    std::vector<DTYPE> input_a_ref(total_elem_num);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&input_a_host), input_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&input_a_device, input_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < total_elem_num; ++i) {
      if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        ((DTYPE *)input_a_host)[i] = DTYPE(std::sin(float(i)));
      } else if (std::is_same<DTYPE, aclFloat16>::value) {
        ((DTYPE *)input_a_host)[i] = aclFloatToFloat16(std::sin(float(i)));
      }
      input_a_ref[i] = ((DTYPE *)input_a_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(input_a_device, input_size, input_a_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *input_b_host;
    uint8_t *input_b_device;
    std::vector<DTYPE> input_b_ref(total_elem_num);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&input_b_host), input_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&input_b_device, input_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < total_elem_num; ++i) {
      if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        ((DTYPE *)input_b_host)[i] = DTYPE(std::cos(float(i)));
      } else if (std::is_same<DTYPE, aclFloat16>::value) {
        ((DTYPE *)input_b_host)[i] = aclFloatToFloat16(std::sin(float(i)));
      }
      input_b_ref[i] = ((DTYPE *)input_b_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(input_b_device, input_size, input_b_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t output_size = total_elem_num * sizeof(DTYPE);
    uint8_t *output_host;
    uint8_t *output_device;
    std::vector<DTYPE> output_ref(total_elem_num);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&output_host), output_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&output_device, output_size, ACL_MEM_MALLOC_HUGE_FIRST));

    if (std::is_same<DTYPE, float>::value) {
      llm_kernels::ascend::InvokeAdd<float>((float *)input_a_device, (float *)input_b_device, nullptr,
                                            (float *)output_device, hidden_units_num, seq_len, stream,
                                            GetTestWorkSpaceFunc);
    } else if (std::is_same<DTYPE, aclFloat16>::value || std::is_same<DTYPE, half_float::half>::value) {
      llm_kernels::ascend::InvokeAdd<aclFloat16>((aclFloat16 *)input_a_device, (aclFloat16 *)input_b_device, nullptr,
                                                 (aclFloat16 *)output_device, hidden_units_num, seq_len, stream,
                                                 GetTestWorkSpaceFunc);
    } else {
      throw std::invalid_argument("Invalid add type type, only support float16 or float32.");
    }

    ACL_CHECK_RET(aclrtSynchronizeStream(stream));
    ACL_CHECK_RET(aclrtMemcpy(output_host, output_size, output_device, output_size, ACL_MEMCPY_DEVICE_TO_HOST));

    for (size_t i = 0; i < total_elem_num; ++i) {
      float ref_value = 0.0f;
      if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        ref_value = float(DTYPE(alpha) * ((DTYPE *)input_a_host)[i] + ((DTYPE *)input_b_host)[i]);
      } else if (std::is_same<DTYPE, aclFloat16>::value) {
        ref_value =
            alpha * aclFloat16ToFloat(((DTYPE *)input_a_host)[i]) + aclFloat16ToFloat(((DTYPE *)input_b_host)[i]);
      } else {
        throw std::invalid_argument("Invalid add type type, only support float16 or float32.");
      }
      EXPECT_NEAR(ref_value, ((DTYPE *)(output_host))[i], 1e-4);
    }

    ACL_CHECK_RET(aclrtFree(output_device));
    ACL_CHECK_RET(aclrtFreeHost(output_host));
    ACL_CHECK_RET(aclrtFree(input_b_device));
    ACL_CHECK_RET(aclrtFreeHost(input_b_host));
    ACL_CHECK_RET(aclrtFree(input_a_device));
    ACL_CHECK_RET(aclrtFreeHost(input_a_host));
  }
};

TEST_F(LlamaAscendAddTestSuit, KernelTest) { RunAddTest<half_float::half>(); }

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels