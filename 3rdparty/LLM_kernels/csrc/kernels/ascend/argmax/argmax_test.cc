/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "aclrtlaunch_InvokeArgmaxFloatKernel.h"
#include "aclrtlaunch_InvokeArgmaxHalfKernel.h"
#include "csrc/kernels/ascend/argmax/argmax.h"
#include "csrc/kernels/ascend/argmax/argmax_kernel.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"
#include "tests/references/argmax.h"

#include "tiling/platform/platform_ascendc.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendArgMaxTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::context;
  using AscendTestSuitBase::default_device;
  using AscendTestSuitBase::is_inited;
  using AscendTestSuitBase::stream;

  template <typename DTYPE>
  void RunArgmaxTest() {
    using OUT_DTYPE = uint32_t;
    ArgmaxConfigTiling argmax_config_tiling;
    argmax_config_tiling.batch_size = 129;
    argmax_config_tiling.vocab_size = 32000;
    argmax_config_tiling.tile_num = 1;
    argmax_config_tiling.block_handle_num =
        (argmax_config_tiling.batch_size + ARGMAX_SINGLE_BLOCK_CAPACITY - 1) / ARGMAX_SINGLE_BLOCK_CAPACITY;
    ArgmaxConfigTiling *buf = &argmax_config_tiling;
    size_t tiling_size = sizeof(ArgmaxConfigTiling);
    uint8_t *tiling_device;
    ACL_CHECK_RET(aclrtMalloc((void **)&tiling_device, tiling_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    ACL_CHECK_RET(aclrtMemcpy(tiling_device, tiling_size, (void *)buf, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t input_size = argmax_config_tiling.batch_size * argmax_config_tiling.vocab_size * sizeof(DTYPE);
    uint8_t *input_host;
    uint8_t *input_device;
    std::vector<DTYPE> input_ref(argmax_config_tiling.batch_size * argmax_config_tiling.vocab_size);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&input_host), input_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&input_device, input_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    for (size_t i = 0; i < argmax_config_tiling.batch_size * argmax_config_tiling.vocab_size; ++i) {
      if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        ((DTYPE *)input_host)[i] = DTYPE(std::sin(float(i)));
      } else if (std::is_same<DTYPE, aclFloat16>::value) {
        ((DTYPE *)input_host)[i] = aclFloatToFloat16(std::sin(float(i)));
      } else {
        throw std::invalid_argument("Invalid embedding lookup type, only support float16 or float32.");
      }
      input_ref[i] = ((DTYPE *)input_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(input_device, input_size, input_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t output_size = argmax_config_tiling.batch_size * sizeof(OUT_DTYPE);
    uint8_t *output_host;
    uint8_t *output_device;
    std::vector<OUT_DTYPE> output_ref(argmax_config_tiling.batch_size);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&output_host), output_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&output_device, output_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));

    if (std::is_same<DTYPE, aclFloat16>::value || std::is_same<DTYPE, half_float::half>::value) {
      ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeArgmaxHalfKernel)(argmax_config_tiling.block_handle_num, stream,
                                                                input_device, output_device, tiling_device));
    } else if (std::is_same<DTYPE, float>::value) {
      ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeArgmaxFloatKernel)(argmax_config_tiling.block_handle_num, stream,
                                                                 input_device, output_device, tiling_device));
    } else {
      throw std::invalid_argument("Invalid argmax data type, only support float16 or float32.");
    }
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));
    ACL_CHECK_RET(aclrtSynchronizeDevice());
    ACL_CHECK_RET(aclrtMemcpy(output_host, output_size, output_device, output_size, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK_RET(aclrtSynchronizeDevice());
    // check correctness
    ArgmaxRef<DTYPE>(input_ref.data(), output_ref.data(), argmax_config_tiling.batch_size,
                     argmax_config_tiling.vocab_size);
    for (size_t idx = 0; idx < argmax_config_tiling.batch_size; ++idx) {
      EXPECT_EQ(output_ref[idx], ((uint32_t *)output_host)[idx]) << "idx:" << idx;
    }

    ACL_CHECK_RET(aclrtFree(output_device));
    ACL_CHECK_RET(aclrtFreeHost(output_host));
    ACL_CHECK_RET(aclrtFree(input_device));
    ACL_CHECK_RET(aclrtFreeHost(input_host));
    ACL_CHECK_RET(aclrtFree(tiling_device));
  }
};

TEST_F(LlamaAscendArgMaxTestSuit, ArgMaxKernelTest) { RunArgmaxTest<half_float::half>(); }

TEST_F(LlamaAscendArgMaxTestSuit, ArgMaxTest) {
  aclTensor *input_tensor = nullptr;
  void *input_workspace = nullptr;
  const std::vector<int64_t> input_shape = {1, 2};

  aclTensor *output_tensor = nullptr;
  void *output_workspace = nullptr;
  const std::vector<int64_t> output_shape = {1, 1};
  CreateAclTensor(input_shape, &input_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &input_tensor);
  CreateAclTensor(output_shape, &output_workspace, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &output_tensor);
  std::vector<half_float::half> input_vec_host(GetShapeSize(input_shape));
  std::vector<half_float::half> out_vec_host(GetShapeSize(output_shape));
  for (size_t i = 0; i < input_vec_host.size(); ++i) {
    input_vec_host[i] = (half_float::half)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  ACL_CHECK_RET(aclrtMemcpyAsync(input_workspace, GetShapeSize(input_shape) * sizeof(half_float::half),
                                 input_vec_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  int64_t arg_max_dim = -1;
  bool arg_max_keepdim = true;
  ArgMax(input_tensor, arg_max_dim, arg_max_keepdim, &output_tensor, stream, llm_kernels::utils::GetTestWorkSpaceFunc);

  ACL_CHECK_RET(aclrtMemcpyAsync(out_vec_host.data(), GetShapeSize(output_shape) * sizeof(int64_t), output_workspace,
                                 GetShapeSize(output_shape) * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  float max_value = std::numeric_limits<float>::min();
  size_t max_idx = 0ul;
  for (size_t i = 0; i < input_vec_host.size(); ++i) {
    if (input_vec_host[i] > max_value) {
      max_idx = i;
      max_value = input_vec_host[i];
    }
  }
  EXPECT_EQ(max_idx, static_cast<size_t>(out_vec_host[0]));

  ACL_CHECK_RET(aclDestroyTensor(output_tensor));
  ACL_CHECK_RET(aclDestroyTensor(input_tensor));
  ACL_CHECK_RET(aclrtFree(input_workspace));
  ACL_CHECK_RET(aclrtFree(output_workspace));
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
