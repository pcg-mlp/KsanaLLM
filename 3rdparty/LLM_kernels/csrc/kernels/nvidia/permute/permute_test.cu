/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <random>
#include <sstream>

#include <gtest/gtest.h>

#include "permute.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {
class LlamaNvidiaPermuteTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
};

TEST_F(LlamaNvidiaPermuteTestSuit, PermuteKernelTest) {
  std::vector<size_t> input_shape = {3, 5, 7, 9};
  std::vector<size_t> permute = {2, 0, 1, 3};
  BufferMeta input = CreateBuffer<float>(MemoryType::MEMORY_GPU, {3, 5, 7, 9}, /*is_random_init*/ true);
  
  BufferMeta output = CreateBuffer<float>(MemoryType::MEMORY_GPU, {7, 3, 5, 9}, /*is_random_init*/ false);
  InvokePermute<4ul, sizeof(float)>(input.data_ptr, output.data_ptr, input_shape, permute, stream);
  BufferMeta output_host = CopyToHost<float>(output);
  BufferMeta input_host = CopyToHost<float>(input);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  float* input_h = reinterpret_cast<float*>(input_host.data_ptr);
  float* output_h = reinterpret_cast<float*>(output_host.data_ptr);
  for (size_t i = 0; i < 20; i++) {
    printf("Input  Num %d = %f\n", i, input_h[i]);
  }
  for (size_t i = 0; i < 20; i++) {
    printf("Output  Num %d = %f\n", i, output_h[i]);
  }
}
}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
