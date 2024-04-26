/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <random>
#include <sstream>

#include <gtest/gtest.h>

#include "3rdparty/half/include/half.hpp"
#include "cast.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {
class LlamaNvidiaCastTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
};

TEST_F(LlamaNvidiaCastTestSuit, ConvertHalfToFloatVectorizeTest) {
  const std::vector<size_t> test_data_size = {1ul, 1024ul, 32768ul};
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<half>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<half>(input);
    BufferMeta output_device = CreateBuffer<float>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ false);
    ConvertHalfToFloatVectorize(reinterpret_cast<float*>(output_device.data_ptr),
                                reinterpret_cast<half*>(input.data_ptr), input_data_size, stream);
    BufferMeta output_host = CopyToHost<float>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      float* output_ptr = reinterpret_cast<float*>(output_host.data_ptr);
      half* output_ref_ptr = reinterpret_cast<half*>(input_host_reference.data_ptr);
      EXPECT_TRUE(output_ptr[idx] == (float)(half_float::half)(output_ref_ptr[idx]));
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, ConvertHalfToFloatTest) {
  const std::vector<size_t> test_data_size = {1ul, 1024ul, 32768ul};
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<half>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<half>(input);
    BufferMeta output_device = CreateBuffer<float>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ false);
    HalfToFloat(reinterpret_cast<half*>(input.data_ptr), input_data_size,
                reinterpret_cast<float*>(output_device.data_ptr), stream);
    BufferMeta output_host = CopyToHost<float>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      float* output_ptr = reinterpret_cast<float*>(output_host.data_ptr);
      half* output_ref_ptr = reinterpret_cast<half*>(input_host_reference.data_ptr);
      EXPECT_TRUE(output_ptr[idx] == (float)(half_float::half)(output_ref_ptr[idx]));
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, ConvertBFloatToFloatTest) {
  const std::vector<size_t> test_data_size = {1ul, 1024ul, 32768ul};
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host_reference = CopyToHost<__nv_bfloat16>(input);
    BufferMeta output_device = CreateBuffer<float>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ false);
    BFloat16ToFloat(reinterpret_cast<__nv_bfloat16*>(input.data_ptr), input_data_size,
                    reinterpret_cast<float*>(output_device.data_ptr), stream);
    BufferMeta output_host = CopyToHost<float>(output_device);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      float* output_ptr = reinterpret_cast<float*>(output_host.data_ptr);
      __nv_bfloat16* output_ref_ptr = reinterpret_cast<__nv_bfloat16*>(input_host_reference.data_ptr);
      EXPECT_TRUE(output_ptr[idx] == (float)(output_ref_ptr[idx]));
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, ConvertBFP16ToFP16Test) {
  const std::vector<size_t> test_data_size = {1ul, 1024ul, 32768ul};
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host = CopyToHost<__nv_bfloat16>(input);
    BFP16ToFP16(input.data_ptr, input_data_size, stream);
    BufferMeta output_host = CopyToHost<half>(input);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      __nv_bfloat16* input_ptr = reinterpret_cast<__nv_bfloat16*>(input_host.data_ptr);
      half* output_ptr = reinterpret_cast<half*>(output_host.data_ptr);
      EXPECT_TRUE((float)input_ptr[idx] == (float)(output_ptr[idx]));
    }
  }
}

TEST_F(LlamaNvidiaCastTestSuit, ConvertFP16ToBFP16Test) {
  const std::vector<size_t> test_data_size = {1ul, 1024ul, 32768ul};
  for (size_t input_data_size : test_data_size) {
    BufferMeta input = CreateBuffer<half>(MemoryType::MEMORY_GPU, {input_data_size}, /*is_random_init*/ true);
    BufferMeta input_host = CopyToHost<half>(input);
    FP16ToBFP16(input.data_ptr, input_data_size, stream);
    BufferMeta output_host = CopyToHost<__nv_bfloat16>(input);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    for (size_t idx = 0ul; idx < input_data_size; ++idx) {
      half* input_ptr = reinterpret_cast<half*>(input_host.data_ptr);
      __nv_bfloat16* output_ptr = reinterpret_cast<__nv_bfloat16*>(output_host.data_ptr);
      EXPECT_TRUE(fabs((float)input_ptr[idx] - (float)(output_ptr[idx])) < 1e-3);
    }
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
