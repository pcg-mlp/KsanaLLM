/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/ascend/assemble_last_token/assemble_last_token.h"

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendAssembleLastTokenTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::context;
  using AscendTestSuitBase::default_device;
  using AscendTestSuitBase::is_inited;
  using AscendTestSuitBase::stream;

  template <typename DTYPE>
  void RunAssembleLastTokenKernelTest() {
    constexpr int32_t hidden_units_num = 4096;
    const std::vector<uint64_t> seq_len_offset = {0, 1, 3, 5};
    int32_t batch_size = seq_len_offset.size() - 1;

    size_t input_size = seq_len_offset.back() * hidden_units_num * sizeof(DTYPE);
    uint8_t *input_host;
    uint8_t *input_device;
    std::vector<DTYPE> input_ref(input_size);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&input_host), input_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&input_device, input_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < seq_len_offset.back() * hidden_units_num; ++i) {
      if (std::is_same<DTYPE, half_float::half>::value || std::is_same<DTYPE, float>::value) {
        ((DTYPE *)input_host)[i] = DTYPE(std::sin(float(i)));
      } else if (std::is_same<DTYPE, aclFloat16>::value) {
        ((DTYPE *)input_host)[i] = aclFloatToFloat16(std::sin(float(i)));
      }
      input_ref[i] = ((DTYPE *)input_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(input_device, input_size, input_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *seq_len_offset_device;
    ACL_CHECK_RET(aclrtMalloc((void **)&seq_len_offset_device, seq_len_offset.size() * sizeof(uint64_t),
                              ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK_RET(aclrtMemcpy(seq_len_offset_device, seq_len_offset.size() * sizeof(uint64_t), seq_len_offset.data(),
                              seq_len_offset.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));

    size_t output_size = batch_size * hidden_units_num * sizeof(DTYPE);
    uint8_t *output_host;
    uint8_t *output_device;
    std::vector<DTYPE> output_ref(output_size);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&output_host), output_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&output_device, output_size, ACL_MEM_MALLOC_HUGE_FIRST));

    if (std::is_same<DTYPE, half_float::half>::value || std::is_same<DTYPE, aclFloat16>::value) {
      llm_kernels::ascend::InvokeAssembleLastToken<aclFloat16>(
          reinterpret_cast<aclFloat16 *>(input_device), reinterpret_cast<size_t *>(seq_len_offset_device), nullptr,
          batch_size, hidden_units_num, reinterpret_cast<aclFloat16 *>(output_device), stream, GetTestWorkSpaceFunc);
    } else if (std::is_same<DTYPE, float>::value) {
      llm_kernels::ascend::InvokeAssembleLastToken<float>(
          reinterpret_cast<float *>(input_device), reinterpret_cast<size_t *>(seq_len_offset_device), nullptr,
          batch_size, hidden_units_num, reinterpret_cast<float *>(output_device), stream, GetTestWorkSpaceFunc);
    } else {
      GTEST_SKIP_("This test is just supported float and float16.");
    }
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));
    ACL_CHECK_RET(aclrtMemcpy(output_host, output_size, output_device, output_size, ACL_MEMCPY_DEVICE_TO_HOST));

    DTYPE* ouput = ((DTYPE*)output_host);
    bool is_break = false;
    for(int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      size_t batch_offset = seq_len_offset[batch_idx] * hidden_units_num;
      size_t batch_seq_len = seq_len_offset[batch_idx + 1] - seq_len_offset[batch_idx];
      for(int32_t hidden_elem_idx = 0; hidden_elem_idx < hidden_units_num; ++hidden_elem_idx) {
        DTYPE ref_val = input_ref[batch_offset + (batch_seq_len - 1) * hidden_units_num + hidden_elem_idx];
        DTYPE val = ouput[batch_idx * hidden_units_num + hidden_elem_idx];
        EXPECT_EQ(ref_val, val) << "[" << batch_idx << ", " << hidden_elem_idx << "] ref_val: " << ref_val << ", val: " << val;
        if (ref_val != val) {
          is_break = true;
          break;
        }
      }
      if (is_break) {
        break;
      }
    }

    ACL_CHECK_RET(aclrtFreeHost(output_host));
    ACL_CHECK_RET(aclrtFree(output_device));
    ACL_CHECK_RET(aclrtFree(seq_len_offset_device));
    ACL_CHECK_RET(aclrtFreeHost(input_host));
    ACL_CHECK_RET(aclrtFree(input_device));
  }
};

TEST_F(LlamaAscendAssembleLastTokenTestSuit, AssembleLastTokenKernelTest) {
  RunAssembleLastTokenKernelTest<half_float::half>();
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels