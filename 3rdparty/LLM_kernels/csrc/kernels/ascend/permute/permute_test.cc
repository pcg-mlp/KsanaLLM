/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "3rdparty/half/include/half.hpp"
#include "csrc/kernels/ascend/permute/permute.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"
#include "tests/references/permute.h"

#include "aclnnop/aclnn_copy.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendPermuteTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::stream;

  template <typename T>
  void RunPermute() {
    std::vector<uint64_t> input_shape = {80, 20, 128};
    std::vector<uint64_t> permute_dims = {1, 2, 0};
    std::vector<uint64_t> output_shape = input_shape;
    for (uint64_t i = 0; i < permute_dims.size(); ++i) {
      output_shape[i] = input_shape[permute_dims[i]];
    }
    size_t total_elem_nums = 1ul;
    for (uint64_t i = 0; i < output_shape.size(); ++i) {
      total_elem_nums *= output_shape[i];
    }

    size_t tensor_size = total_elem_nums * sizeof(T);

    std::vector<T> input_ref(total_elem_nums);
    uint8_t* input_host;
    uint8_t* input_device;
    ACL_CHECK_RET(aclrtMallocHost((void**)(&input_host), tensor_size));
    ACL_CHECK_RET(aclrtMalloc((void**)&input_device, tensor_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (int i = 0; i < total_elem_nums; ++i) {
      if (std::is_same<T, aclFloat16>::value) {
        ((T*)input_host)[i] = aclFloatToFloat16(float(std::cos(i)));
      } else if (std::is_same<T, float>::value || std::is_same<T, half_float::half>::value) {
        ((T*)input_host)[i] = T(std::sin(i));
      } else {
        throw std::invalid_argument("Invalid permute compute type, only support float16 or float32.");
      }
      input_ref[i] = ((T*)input_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(input_device, tensor_size, input_host, tensor_size, ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<T> output_ref_tmp(total_elem_nums);
    std::vector<T> output_ref(total_elem_nums);
    uint8_t* output_host;
    uint8_t* output_device;
    ACL_CHECK_RET(aclrtMallocHost((void**)(&output_host), tensor_size));
    ACL_CHECK_RET(aclrtMalloc((void**)&output_device, tensor_size, ACL_MEM_MALLOC_HUGE_FIRST));

    if (std::is_same<T, aclFloat16>::value || std::is_same<T, half_float::half>::value) {
      PermuteKernelWrapper<aclFloat16> permute;
      permute.Forward(output_device, input_device, input_shape, permute_dims, stream);
    } else {
      PermuteKernelWrapper<float> permute;
      permute.Forward(output_device, input_device, input_shape, permute_dims, stream);
    }
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));
    ACL_CHECK_RET(aclrtMemcpy(output_host, tensor_size, output_device, tensor_size, ACL_MEMCPY_DEVICE_TO_HOST));

    // permute twice and input ref should same as input_ref output_ref
    RunPermuteRef<T>((void*)(input_ref.data()), (void*)(output_ref_tmp.data()), input_shape, output_shape,
                     permute_dims);

    for (size_t idx = 0; idx < total_elem_nums; ++idx) {
      if (std::is_same<T, aclFloat16>::value) {
        EXPECT_NEAR(aclFloat16ToFloat(output_ref_tmp[idx]), aclFloat16ToFloat(((T*)output_host)[idx]), 1e-2);
      } else if (std::is_same<T, float>::value || std::is_same<T, half_float::half>::value) {
        EXPECT_NEAR(output_ref_tmp[idx], ((T*)output_host)[idx], 1e-2);
      } else {
        throw std::invalid_argument("Invalid permute compute type, only support float16 or float32.");
      }
    }

    ACL_CHECK_RET(aclrtFree(output_device));
    ACL_CHECK_RET(aclrtFreeHost(output_host));
    ACL_CHECK_RET(aclrtFree(input_device));
    ACL_CHECK_RET(aclrtFreeHost(input_host));
  }
};

TEST_F(LlamaAscendPermuteTestSuit, CommonTest) {
  aclDataType dtype = aclDataType::ACL_FLOAT16;
  uint64_t workspace_size = 0ul;
  void* workspace_ptr = nullptr;

  aclTensor* input_tensor = nullptr;
  void* input_workspace = nullptr;
  const std::vector<int64_t> input_shape = {3, 4, 5};
  CreateAclTensor(input_shape, &input_workspace, dtype, aclFormat::ACL_FORMAT_ND, &input_tensor);

  const std::vector<int64_t> dims = {2, 0, 1};
  aclTensor* output_tensor = nullptr;
  Permute(input_tensor, &input_workspace, &output_tensor, dims, stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  aclDataType output_dtype;
  ACL_CHECK_RET(aclGetDataType(output_tensor, &output_dtype));
  EXPECT_EQ(output_dtype, dtype);

  int64_t* output_t_shape_ptr = nullptr;
  uint64_t output_t_dims_num = 0;
  ACL_CHECK_RET(aclGetViewShape(output_tensor, &output_t_shape_ptr, &output_t_dims_num));
  for (uint64_t i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(output_t_shape_ptr[i], input_shape[dims[i]]);
  }

  if (workspace_size > 0 && workspace_ptr != nullptr) {
    ACL_CHECK_RET(aclrtFree(workspace_ptr));
  }
  ACL_CHECK_RET(aclDestroyTensor(output_tensor));
  ACL_CHECK_RET(aclDestroyTensor(input_tensor));
  ACL_CHECK_RET(aclrtFree(input_workspace));
}

TEST_F(LlamaAscendPermuteTestSuit, PermuteKernelTest) { RunPermute<half_float::half>(); }

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
