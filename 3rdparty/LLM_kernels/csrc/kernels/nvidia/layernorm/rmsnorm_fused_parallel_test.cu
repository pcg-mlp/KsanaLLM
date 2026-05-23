/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/layernorm/layernorm.h"
#include "csrc/kernels/nvidia/layernorm/rmsnorm_fused_parallel.h"

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class ParallelRmsNormTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  template <typename T>
  void TestParallelRmsNorm(int batch_size, int dim1, int dim2, bool is_perf) {
    const float eps = 1e-6f;
    const auto bs = static_cast<size_t>(batch_size);
    const auto d1 = static_cast<size_t>(dim1);
    const auto d2 = static_cast<size_t>(dim2);

    // Allocate buffers
    BufferMeta input1 = CreateBuffer<T>(MemoryType::MEMORY_GPU, {bs, d1}, /*is_random_init*/ true);
    BufferMeta weight1 = CreateBuffer<T>(MemoryType::MEMORY_GPU, {d1}, /*is_random_init*/ true);
    BufferMeta input2 = CreateBuffer<T>(MemoryType::MEMORY_GPU, {bs, d2}, /*is_random_init*/ true);
    BufferMeta weight2 = CreateBuffer<T>(MemoryType::MEMORY_GPU, {d2}, /*is_random_init*/ true);
    BufferMeta fused_out1 = CreateBuffer<T>(MemoryType::MEMORY_GPU, {bs, d1}, /*is_random_init*/ false);
    BufferMeta fused_out2 = CreateBuffer<T>(MemoryType::MEMORY_GPU, {bs, d2}, /*is_random_init*/ false);
    BufferMeta ref_out1 = CreateBuffer<T>(MemoryType::MEMORY_GPU, {bs, d1}, /*is_random_init*/ false);
    BufferMeta ref_out2 = CreateBuffer<T>(MemoryType::MEMORY_GPU, {bs, d2}, /*is_random_init*/ false);

    // Run fused kernel
    InvokeParallelRmsNorm<T>(static_cast<const T*>(input1.data_ptr), static_cast<const T*>(weight1.data_ptr),
                             static_cast<T*>(fused_out1.data_ptr), dim1, /*stride_input1*/ dim1,
                             static_cast<const T*>(input2.data_ptr), static_cast<const T*>(weight2.data_ptr),
                             static_cast<T*>(fused_out2.data_ptr), dim2, /*stride_input2*/ dim2, batch_size, eps,
                             stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Run reference
    InvokeRMSNorm<T>(static_cast<T*>(ref_out1.data_ptr), static_cast<T*>(input1.data_ptr),
                     static_cast<T*>(weight1.data_ptr), eps, batch_size, dim1, stream);
    InvokeRMSNorm<T>(static_cast<T*>(ref_out2.data_ptr), static_cast<T*>(input2.data_ptr),
                     static_cast<T*>(weight2.data_ptr), eps, batch_size, dim2, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Correctness check (skip when benchmarking)
    if (!is_perf) {
      const float atol = std::is_same<T, float>::value ? 1e-5f : 1e-3f;
      const float rtol = atol;
      std::string type_str = std::is_same<T, half>::value ? "half" : "bfloat16";
      if (std::is_same<T, float>::value) type_str = "float";
      const std::string tag =
          type_str + "_b" + std::to_string(batch_size) + "_d" + std::to_string(dim1) + "_d" + std::to_string(dim2);

      EXPECT_TRUE(CheckResult<T>("out1_" + tag, fused_out1, ref_out1, atol, rtol)) << "Output1 mismatch: " << tag;
      EXPECT_TRUE(CheckResult<T>("out2_" + tag, fused_out2, ref_out2, atol, rtol)) << "Output2 mismatch: " << tag;
    }

    // Performance benchmark (only when is_perf)
    if (is_perf) {
      const int warmups = 5;
      const int iterations = 20;

      auto fused_run = [&]() {
        InvokeParallelRmsNorm<T>(static_cast<const T*>(input1.data_ptr), static_cast<const T*>(weight1.data_ptr),
                                 static_cast<T*>(fused_out1.data_ptr), dim1, dim1,
                                 static_cast<const T*>(input2.data_ptr), static_cast<const T*>(weight2.data_ptr),
                                 static_cast<T*>(fused_out2.data_ptr), dim2, dim2, batch_size, eps, stream);
      };
      auto separate_run = [&]() {
        InvokeRMSNorm<T>(static_cast<T*>(ref_out1.data_ptr), static_cast<T*>(input1.data_ptr),
                         static_cast<T*>(weight1.data_ptr), eps, batch_size, dim1, stream);
        InvokeRMSNorm<T>(static_cast<T*>(ref_out2.data_ptr), static_cast<T*>(input2.data_ptr),
                         static_cast<T*>(weight2.data_ptr), eps, batch_size, dim2, stream);
      };

      const float fused_ms = MeasureCudaExecutionTime(fused_run, stream, warmups, iterations);
      const float separate_ms = MeasureCudaExecutionTime(separate_run, stream, warmups, iterations);
      const float speedup = separate_ms / fused_ms;

      std::cout << std::left << std::setw(10) << batch_size << std::setw(8) << dim1 << std::setw(8) << dim2
                << std::fixed << std::setprecision(4) << std::setw(14) << fused_ms << std::setw(14) << separate_ms
                << std::setprecision(2) << std::setw(10) << speedup << "x" << std::endl;
    }

    // Cleanup
    DeleteBuffer(ref_out2);
    DeleteBuffer(ref_out1);
    DeleteBuffer(fused_out2);
    DeleteBuffer(fused_out1);
    DeleteBuffer(weight2);
    DeleteBuffer(input2);
    DeleteBuffer(weight1);
    DeleteBuffer(input1);
  }
};

// ============================================================================
// Correctness Tests
// ============================================================================

TEST_F(ParallelRmsNormTestSuit, BFloat16CorrectnessTest) {
  const std::vector<std::tuple<int, int, int>> configs = {
      {1, 1024, 512}, {1, 1536, 512}, {4, 2048, 512}, {32, 1536, 512}, {32, 7168, 2048}, {128, 4096, 1024},
  };
  for (const auto& [bs, d1, d2] : configs) {
    TestParallelRmsNorm<__nv_bfloat16>(bs, d1, d2, /*is_perf*/ false);
  }
}

// ============================================================================
// Performance Tests (disabled by default)
// ============================================================================

TEST_F(ParallelRmsNormTestSuit, DISABLED_BFloat16PerformanceTest) {
  const std::vector<std::tuple<int, int, int>> configs = {
      {1, 1024, 512}, {8, 1024, 512}, {32, 1024, 512}, {128, 1024, 512}, {256, 1024, 512}, {1024, 1024, 512},
      {1, 1536, 512}, {8, 1536, 512}, {32, 1536, 512}, {128, 1536, 512}, {256, 1536, 512}, {1024, 1536, 512}};
  std::cout << "\n=== ParallelRmsNorm Performance (bfloat16) ===" << std::endl;
  std::cout << std::left << std::setw(10) << "batch" << std::setw(8) << "dim1" << std::setw(8) << "dim2"
            << std::setw(14) << "fused(ms)" << std::setw(14) << "separate(ms)" << std::setw(10) << "speedup"
            << std::endl;
  std::cout << std::string(64, '-') << std::endl;
  for (const auto& [bs, d1, d2] : configs) {
    TestParallelRmsNorm<__nv_bfloat16>(bs, d1, d2, /*is_perf*/ true);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
