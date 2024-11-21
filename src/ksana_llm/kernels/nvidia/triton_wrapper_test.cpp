/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "test.h"

#include "ksana_llm/kernels/nvidia/triton_wrapper.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/tensor.h"

namespace fs = std::filesystem;

namespace ksana_llm {

class TritonWrapperTest : public testing::Test {
 public:
  void SetUp() override {
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

 protected:
  int32_t device{-1};
  cudaStream_t stream;
};

TEST_F(TritonWrapperTest, CompileTritonCodeAndRun) {
  int cuda_ver;
  CUDA_CHECK(cudaRuntimeGetVersion(&cuda_ver));
  if (cuda_ver <= 11080) {
    std::string err_msg = "Triton is note stable in CUDA Version: " + std::to_string(cuda_ver);
    GTEST_SKIP_(err_msg.c_str());
  }

  int status_code = std::system("python /tmp/triton_wrapper_test.py");
  if (status_code != 0) {
    GTEST_SKIP_("Compile triton code and run test is skipped.");
  }
  uint32_t size = 98432;
  uint32_t block_size = 1024;
  TritonKernel kernel;
  kernel.shm_size = 0;
  kernel.grid_x = (size + block_size - 1) / block_size;
  kernel.grid_y = 1;
  kernel.num_warps = 4;
  kernel.kernel_name = "add_kernel_0d1d2d3d";
  dim3 grid(kernel.grid_x, 1, 1);
  dim3 block(kernel.num_warps * 32, 1, 1);
  float* x_ptr;
  float* y_ptr;
  float* output_ptr;
  std::vector<float> x_vec(size, 0.f);
  std::vector<float> y_vec(size, 0.f);
  std::vector<float> out_vec(size, 0.f);
  CUDA_CHECK(cudaMalloc(&x_ptr, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&y_ptr, size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(x_ptr, x_vec.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(y_ptr, y_vec.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&output_ptr, size * sizeof(float)));
  void* args[4] = {&x_ptr, &y_ptr, &output_ptr, &size};
  if (LoadTritonKernelFromFile("./triton_wrapper_test_kernel.ptx", kernel) != CUDA_SUCCESS) {
    GTEST_SKIP_("Compile triton code and run test is skipped for compile toolchain incompatible.");
  }
  ASSERT_TRUE(InvokeTritonKernel(kernel, grid, block, args, stream, nullptr) == CUDA_SUCCESS);
  CUDA_CHECK(cudaMemcpy(out_vec.data(), output_ptr, size * sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t idx = 0; idx < size; ++idx) {
    EXPECT_EQ(out_vec[idx], x_vec[idx] + y_vec[idx]);
  }
  CUDA_CHECK(cudaFree(x_ptr));
  CUDA_CHECK(cudaFree(y_ptr));
  CUDA_CHECK(cudaFree(output_ptr));
}

}  // namespace ksana_llm