/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "csrc/kernels/ascend/permute/permute.h"
#include "csrc/kernels/ascend/transpose/transpose.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

#ifdef ENABLE_ACL_ATB
#  include "csrc/utils/ascend/atb_executor.h"
#endif

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendTransposeTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::context;
  using AscendTestSuitBase::default_device;
  using AscendTestSuitBase::is_inited;
  using AscendTestSuitBase::stream;

#ifdef ENABLE_ACL_ATB
  template <typename DTYPE>
  void RunATBReshapeAndCacheTest() {
    aclDataType aclnn_dtype = aclDataType::ACL_FLOAT16;
    if (std::is_same<DTYPE, float>::value) {
      aclnn_dtype = aclDataType::ACL_FLOAT;
    } else if (std::is_same<DTYPE, aclFloat16>::value || std::is_same<DTYPE, half_float::half>::value) {
      aclnn_dtype = aclDataType::ACL_FLOAT16;
    } else {
      GTEST_SKIP_("This test is just supported float and float16.");
    }
    const int32_t num_tokens{512};
    const int32_t num_kv_heads{40};
    const int32_t num_heads{40};
    const int32_t head_dim{128};
    const int32_t num_blocks{32};
    const int32_t block_size{16};
    // setting key input
    size_t key_elem_nums = num_tokens * num_kv_heads * head_dim;
    size_t key_size = key_elem_nums * sizeof(DTYPE);
    void* key_host;
    void* key_device;
    std::vector<DTYPE> key_ref(key_elem_nums);
    ACL_CHECK_RET(aclrtMallocHost(&key_host, key_size));
    ACL_CHECK_RET(aclrtMalloc(&key_device, key_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < key_elem_nums; ++i) {
      if (std::is_same<DTYPE, aclFloat16>::value) {
        reinterpret_cast<DTYPE*>(key_host)[i] = aclFloatToFloat16(float(std::sin(i)));
      } else if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        reinterpret_cast<DTYPE*>(key_host)[i] = DTYPE(std::sin(i));
      } else {
        throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
      }
      key_ref[i] = reinterpret_cast<DTYPE*>(key_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(key_device, key_size, key_host, key_size, ACL_MEMCPY_HOST_TO_DEVICE));
    // setting value input
    size_t value_elem_nums = num_tokens * num_kv_heads * head_dim;
    size_t value_size = key_elem_nums * sizeof(DTYPE);
    void* value_host;
    void* value_device;
    std::vector<DTYPE> value_ref(value_elem_nums);
    ACL_CHECK_RET(aclrtMallocHost(&value_host, key_size));
    ACL_CHECK_RET(aclrtMalloc(&value_device, value_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < value_elem_nums; ++i) {
      if (std::is_same<DTYPE, aclFloat16>::value) {
        reinterpret_cast<DTYPE*>(value_host)[i] = aclFloatToFloat16(float(std::cos(i)));
      } else if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        reinterpret_cast<DTYPE*>(value_host)[i] = DTYPE(std::cos(i));
      } else {
        throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
      }
      value_ref[i] = reinterpret_cast<DTYPE*>(value_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(value_device, value_size, value_host, value_size, ACL_MEMCPY_HOST_TO_DEVICE));
    // slot mapping
    void* slot_mapping_device;
    std::vector<int32_t> slot_mapping_ref(num_tokens, 0);
    ACL_CHECK_RET(aclrtMalloc(&slot_mapping_device, num_tokens * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    std::iota(slot_mapping_ref.begin(), slot_mapping_ref.end(), 1);
    ACL_CHECK_RET(aclrtMemcpy(slot_mapping_device, num_tokens * sizeof(int32_t), slot_mapping_ref.data(),
                              num_tokens * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));
    // k cache
    size_t k_cache_elem_nums = num_blocks * block_size * num_kv_heads * head_dim;
    void* k_cache_device;
    ACL_CHECK_RET(aclrtMalloc(&k_cache_device, k_cache_elem_nums * sizeof(DTYPE), ACL_MEM_MALLOC_HUGE_FIRST));
    // v cache
    size_t v_cache_elem_nums = num_blocks * block_size * num_kv_heads * head_dim;
    void* v_cache_device;
    ACL_CHECK_RET(aclrtMalloc(&v_cache_device, v_cache_elem_nums * sizeof(DTYPE), ACL_MEM_MALLOC_HUGE_FIRST));

    atb::infer::ReshapeAndCacheParam op_param;
    llm_kernels::utils::ATBOperationExecutor atb_op_executor;
    atb_op_executor.Init(default_device, op_param);
    atb_op_executor.ResetVariantPack();

    atb_op_executor.SetInputTensor(key_device, {num_tokens, num_kv_heads, head_dim}, aclnn_dtype);
    atb_op_executor.SetInputTensor(value_device, {num_tokens, num_kv_heads, head_dim}, aclnn_dtype);
    atb_op_executor.SetInputTensor(k_cache_device, {num_blocks, block_size, num_kv_heads, head_dim}, aclnn_dtype);
    atb_op_executor.SetInputTensor(v_cache_device, {num_blocks, block_size, num_kv_heads, head_dim}, aclnn_dtype);
    atb_op_executor.SetInputTensor(slot_mapping_device, {num_tokens}, aclDataType::ACL_INT32);
    atb_op_executor.SetOutputTensor(k_cache_device, {num_blocks, block_size, num_kv_heads, head_dim}, aclnn_dtype);
    atb_op_executor.SetOutputTensor(v_cache_device, {num_blocks, block_size, num_kv_heads, head_dim}, aclnn_dtype);
    atb_op_executor.Run(atb_context, llm_kernels::utils::GetTestWorkSpaceFunc);
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));

    ACL_CHECK_RET(aclrtFree(v_cache_device));
    ACL_CHECK_RET(aclrtFree(k_cache_device));
    ACL_CHECK_RET(aclrtFree(slot_mapping_device));
    ACL_CHECK_RET(aclrtFree(value_device));
    ACL_CHECK_RET(aclrtFree(key_device));
  }
#endif
};

TEST_F(LlamaAscendTransposeTestSuit, TransposeTest) {
  aclTensor* input_tensor = nullptr;
  void* input_workspace = nullptr;
  const std::vector<int64_t> input_shape = {3, 2, 1};
  const std::vector<int64_t> dims = {0, 1, 2};
  const std::vector<int64_t> output_shape = {3, 2, 1};
  CreateAclTensor(input_shape, &input_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &input_tensor);
  aclTensor* output_tensor = nullptr;
  std::vector<half_float::half> input_vec_host(GetShapeSize(input_shape));
  std::vector<half_float::half> out_vec_host(GetShapeSize(input_shape));
  for (size_t i = 0; i < input_vec_host.size(); ++i) {
    input_vec_host[i] = (half_float::half)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  ACL_CHECK_RET(aclrtMemcpyAsync(input_workspace, GetShapeSize(input_shape) * sizeof(half_float::half),
                                 input_vec_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  Permute(input_tensor, &input_workspace, &output_tensor, dims, stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  Transpose(input_tensor, &output_tensor, stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  ACL_CHECK_RET(aclrtMemcpyAsync(out_vec_host.data(), GetShapeSize(output_shape) * sizeof(half_float::half),
                                 input_workspace, GetShapeSize(output_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  EXPECT_NEAR(input_vec_host[0], float(out_vec_host[0]), 1e-5);
  for (size_t i = 0; i < input_vec_host.size(); ++i) {
    EXPECT_NEAR(float(input_vec_host[i]), float(out_vec_host[i]), 1e-5);
  }

  ACL_CHECK_RET(aclDestroyTensor(output_tensor));
  ACL_CHECK_RET(aclDestroyTensor(input_tensor));
  ACL_CHECK_RET(aclrtFree(input_workspace));
}

TEST_F(LlamaAscendTransposeTestSuit, CopyTest) {
  aclTensor* input_tensor = nullptr;
  void* input_workspace = nullptr;
  const std::vector<int64_t> input_shape = {1, 2};
  aclTensor* output_tensor = nullptr;
  void* output_workspace = nullptr;
  const std::vector<int64_t> output_shape = {1, 2};
  CreateAclTensor(input_shape, &input_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &input_tensor);
  CreateAclTensor(output_shape, &output_workspace, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &output_tensor);

  std::vector<half_float::half> input_vec_host(GetShapeSize(input_shape));
  std::vector<half_float::half> out_vec_host(GetShapeSize(input_shape));
  for (size_t i = 0; i < input_vec_host.size(); ++i) {
    input_vec_host[i] = (half_float::half)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  ACL_CHECK_RET(aclrtMemcpyAsync(input_workspace, GetShapeSize(input_shape) * sizeof(half_float::half),
                                 input_vec_host.data(), GetShapeSize(input_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  Transpose(input_tensor, &output_tensor, stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  ACL_CHECK_RET(aclrtMemcpyAsync(out_vec_host.data(), GetShapeSize(output_shape) * sizeof(half_float::half),
                                 output_workspace, GetShapeSize(output_shape) * sizeof(half_float::half),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  for (size_t i = 0; i < input_vec_host.size(); ++i) {
    EXPECT_NEAR(float(input_vec_host[i]), float(out_vec_host[i]), 1e-5);
  }

  ACL_CHECK_RET(aclDestroyTensor(output_tensor));
  ACL_CHECK_RET(aclDestroyTensor(input_tensor));
  ACL_CHECK_RET(aclrtFree(input_workspace));
  ACL_CHECK_RET(aclrtFree(output_workspace));
}

#ifdef ENABLE_ACL_ATB
TEST_F(LlamaAscendTransposeTestSuit, TestReshapeAndCache) {
  RunATBReshapeAndCacheTest<half_float::half>();
}
#endif

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
