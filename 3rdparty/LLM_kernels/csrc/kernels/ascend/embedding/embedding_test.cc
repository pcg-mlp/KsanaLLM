/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include <vector>

#include "3rdparty/half/include/half.hpp"
#include "aclrtlaunch_InvokeLookupEmbeddingFloatKernel.h"
#include "aclrtlaunch_InvokeLookupEmbeddingHalfKernel.h"
#include "csrc/kernels/ascend/embedding/embedding.h"
#include "csrc/utils/ascend/common.h"
#include "embedding_kernel.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

#include "aclnnop/aclnn_copy.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class AscendEmbeddingTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::stream;

  template <typename DTYPE>
  void RunEmbeddingTest() {
    EmbeddingConfigTiling emb_config_tiling;
    emb_config_tiling.total_seq_len = 128;
    emb_config_tiling.hidden_units = 4096;
    emb_config_tiling.vocab_size = 32000;
    emb_config_tiling.vocab_id = 0;
    emb_config_tiling.batch_size = 1;
    emb_config_tiling.tile_num = 1;
    emb_config_tiling.start_step = 1;
    EmbeddingConfigTiling *buf = &emb_config_tiling;
    size_t tiling_size = sizeof(EmbeddingConfigTiling);
    uint8_t *tiling_device;
    ACL_CHECK_RET(aclrtMalloc((void **)&tiling_device, tiling_size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK_RET(aclrtMemcpy(tiling_device, tiling_size, (void *)buf, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t input_size = emb_config_tiling.total_seq_len * sizeof(int32_t);
    uint8_t *input_host;
    uint8_t *input_device;
    std::vector<int32_t> input_ref(emb_config_tiling.total_seq_len);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&input_host), input_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&input_device, input_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < emb_config_tiling.total_seq_len; ++i) {
      ((int32_t *)input_host)[i] = static_cast<int32_t>(i);
      input_ref[i] = ((int32_t *)input_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(input_device, input_size, input_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t emb_size = emb_config_tiling.vocab_size * emb_config_tiling.hidden_units * sizeof(DTYPE);
    uint8_t *emb_host;
    uint8_t *emb_device;
    std::vector<DTYPE> emb_ref(emb_config_tiling.vocab_size * emb_config_tiling.hidden_units);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&emb_host), emb_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&emb_device, emb_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < emb_config_tiling.vocab_size * emb_config_tiling.hidden_units; ++i) {
      ((DTYPE *)emb_host)[i] = DTYPE(std::sin(float(i)));
      emb_ref[i] = ((DTYPE *)emb_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(emb_device, emb_size, emb_host, emb_size, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t output_size = emb_config_tiling.total_seq_len * emb_config_tiling.hidden_units * sizeof(DTYPE);
    uint8_t *output_host;
    uint8_t *output_device;
    std::vector<DTYPE> output_ref(emb_config_tiling.total_seq_len * emb_config_tiling.hidden_units);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&output_host), output_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&output_device, output_size, ACL_MEM_MALLOC_HUGE_FIRST));

    if (std::is_same<DTYPE, aclFloat16>::value || std::is_same<DTYPE, half_float::half>::value) {
      ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeLookupEmbeddingHalfKernel)(
          emb_config_tiling.total_seq_len, stream, input_device, emb_device, output_device, tiling_device));
    } else if (std::is_same<DTYPE, float>::value) {
      ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeLookupEmbeddingFloatKernel)(
          emb_config_tiling.total_seq_len, stream, input_device, emb_device, output_device, tiling_device));
    } else {
      throw std::invalid_argument("Invalid embedding lookup type, only support float16 or float32.");
    }
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));

    ACL_CHECK_RET(aclrtMemcpy(output_host, output_size, output_device, output_size, ACL_MEMCPY_DEVICE_TO_HOST));

    // compute ref
    for (uint32_t token_idx = 0; token_idx < emb_config_tiling.total_seq_len; ++token_idx) {
      int32_t token_id = input_ref[token_idx];
      for (uint32_t emb_idx = 0; emb_idx < emb_config_tiling.hidden_units; ++emb_idx) {
        output_ref[token_idx * emb_config_tiling.hidden_units + emb_idx] =
            emb_ref[token_id * emb_config_tiling.hidden_units + emb_idx];
      }
    }

    // check correctness
    for (size_t idx = 0; idx < emb_config_tiling.total_seq_len * emb_config_tiling.hidden_units; ++idx) {
      if (std::is_same<DTYPE, aclFloat16>::value) {
        EXPECT_NEAR(aclFloat16ToFloat(output_host[idx]), aclFloat16ToFloat(output_ref[idx]), 1e-4);
      } else if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        EXPECT_NEAR(output_ref[idx], ((DTYPE *)output_host)[idx], 1e-4);
      } else {
        throw std::invalid_argument("Invalid embedding lookup type, only support float16 or float32.");
      }
    }

    ACL_CHECK_RET(aclrtFree(tiling_device));
    ACL_CHECK_RET(aclrtFree(output_device));
    ACL_CHECK_RET(aclrtFreeHost(output_host));
    ACL_CHECK_RET(aclrtFree(emb_device));
    ACL_CHECK_RET(aclrtFreeHost(emb_host));
    ACL_CHECK_RET(aclrtFree(input_device));
    ACL_CHECK_RET(aclrtFreeHost(input_host));
  }
};

TEST_F(AscendEmbeddingTestSuit, EmbeddingLookupKernelTest) { RunEmbeddingTest<half_float::half>(); }

template <typename T>
int CreateAclTensor(const std::vector<T> &data, const std::vector<int64_t> &shape, void **addr, aclDataType dtype,
                    aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  ACL_CHECK_RET(aclrtMalloc(addr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(*addr, size, data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *addr);
  return 0;
}

TEST_F(AscendEmbeddingTestSuit, EmbeddingLookupTest) {
  // Create weight [100, 16]
  std::vector<float> weights;
  weights.reserve(100 * 16);
  for (size_t i = 0; i < 100; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      weights.push_back(i + (0.01 * j));
    }
  }
  std::vector<int64_t> weight_shape = {100, 16};

  // Create input ids.
  std::vector<int64_t> input_shape = {4};
  std::vector<int64_t> inputs = {8, 48, 68, 88};

  // Create output.
  std::vector<int64_t> out_shape = {4, 16};
  std::vector<float> outs(4 * 16, 0);

  // device addr.
  void *weight_dev_addr = nullptr;
  void *input_dev_addr = nullptr;
  void *out_dev_addr = nullptr;

  // Create tensors.
  aclTensor *weight_tensor = nullptr;
  aclTensor *input_tensor = nullptr;
  aclTensor *out_tensor = nullptr;

  ACL_CHECK_RET(CreateAclTensor(weights, weight_shape, &weight_dev_addr, aclDataType::ACL_FLOAT, &weight_tensor));
  ACL_CHECK_RET(CreateAclTensor(inputs, input_shape, &input_dev_addr, aclDataType::ACL_INT64, &input_tensor));
  ACL_CHECK_RET(CreateAclTensor(outs, out_shape, &out_dev_addr, aclDataType::ACL_FLOAT, &out_tensor));

  LookupEmbedding(input_tensor, weight_tensor, nullptr, out_tensor, stream, llm_kernels::utils::GetTestWorkSpaceFunc);
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  // Check result
  std::vector<float> result(4 * 16, 0);
  ACL_CHECK_RET(aclrtMemcpy(result.data(), result.size() * sizeof(float), out_dev_addr, 4 * 16 * sizeof(float),
                            ACL_MEMCPY_DEVICE_TO_HOST));

  size_t idx = 0;
  for (auto i : inputs) {
    for (size_t j = 0; j < 16; ++j) {
      EXPECT_FLOAT_EQ(i + (0.01 * j), result[idx++]);
    }
  }
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
