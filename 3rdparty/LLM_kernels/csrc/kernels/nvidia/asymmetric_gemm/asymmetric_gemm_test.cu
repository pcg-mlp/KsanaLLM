/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/fpA_intB_gemm/fpA_intB_gemm_template.h"

#include "csrc/kernels/nvidia/weight_only_batched_gemv/kernelLauncher.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

bool IsConfigValid(llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig& config, size_t k) {
  if (config.stages >= 5) {
    return false;
  }
  if (config.split_k_style != llm_kernels::nvidia::cutlass_extensions::SplitKStyle::NO_SPLIT_K) {
    int k_size = (k + config.split_k_factor - 1) / config.split_k_factor;
    if (k_size % 64) {
      return false;
    }
  }
  return true;
}

class NvidiaAsymmetricGemmTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    NvidiaTestSuitBase::SetUp();
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
  }

  void TearDown() override {
    NvidiaTestSuitBase::TearDown();
    cublasDestroy(cublas_handle);
    cublasLtDestroy(cublaslt_handle);
  }

 protected:
  using NvidiaTestSuitBase::stream;
  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;

 protected:
  void TestAsymmetricGemm(const size_t m, const size_t n, const size_t k, const size_t groupsize) {
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    BufferMeta buffer_cutlass_output = CreateBuffer<half>(MemoryType::MEMORY_GPU, {m, n}, false);
    BufferMeta buffer_cuda_output = CreateBuffer<half>(MemoryType::MEMORY_GPU, {m, n}, false);

    BufferMeta buffer_input = CreateBuffer<half>(MemoryType::MEMORY_GPU, {m, k}, true);
    BufferMeta buffer_qweight = CreateBuffer<char>(MemoryType::MEMORY_GPU, {k, n / 2}, true);
    BufferMeta buffer_weight_scales = CreateBuffer<half>(MemoryType::MEMORY_GPU, {k / groupsize, n}, true);

    auto cutlass_gemm = std::make_shared<llm_kernels::nvidia::CutlassFpAIntBGemmRunner<
        half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();

    const size_t ws_bytes = cutlass_gemm->getWorkspaceSize(m, n, k);
    BufferMeta buffer_ws = CreateBuffer<char>(MemoryType::MEMORY_GPU, {ws_bytes}, false);

    int best_config_index = 0;
    {
      float fast_time = std::numeric_limits<float>::max();
      auto configs = cutlass_gemm->getConfigs();
      for (size_t config_index = 0; config_index < configs.size(); config_index++) {
        auto& config = configs[config_index];
        if (!IsConfigValid(config, k)) {
          continue;
        }
        for (size_t i = 0; i < 10; ++i) {
          cutlass_gemm->gemm(reinterpret_cast<const half*>(buffer_input.data_ptr),
                             reinterpret_cast<const cutlass::uint4b_t*>(buffer_qweight.data_ptr),
                             reinterpret_cast<const half*>(buffer_weight_scales.data_ptr),
                             nullptr,  // no zeros
                             nullptr,  // no bias
                             reinterpret_cast<half*>(buffer_cutlass_output.data_ptr), m, n, k, groupsize,
                             cutlass_gemm->getConfigs()[config_index], reinterpret_cast<char*>(buffer_ws.data_ptr),
                             ws_bytes, stream);
        }
        cudaEventRecord(begin, stream);
        for (size_t i = 0; i < 100; ++i) {
          cutlass_gemm->gemm(reinterpret_cast<const half*>(buffer_input.data_ptr),
                             reinterpret_cast<const cutlass::uint4b_t*>(buffer_qweight.data_ptr),
                             reinterpret_cast<const half*>(buffer_weight_scales.data_ptr),
                             nullptr,  // no zeros
                             nullptr,  // no bias
                             reinterpret_cast<half*>(buffer_cutlass_output.data_ptr), m, n, k, groupsize,
                             cutlass_gemm->getConfigs()[config_index], reinterpret_cast<char*>(buffer_ws.data_ptr),
                             ws_bytes, stream);
        }
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        float time;
        cudaEventElapsedTime(&time, begin, end);
        if (time < fast_time) {
          fast_time = time;
          best_config_index = config_index;
        }
      }
    }

    for (size_t i = 0; i < 10; ++i) {
      cutlass_gemm->gemm(reinterpret_cast<const half*>(buffer_input.data_ptr),
                         reinterpret_cast<const cutlass::uint4b_t*>(buffer_qweight.data_ptr),
                         reinterpret_cast<const half*>(buffer_weight_scales.data_ptr),
                         nullptr,  // no zeros
                         nullptr,  // no bias
                         reinterpret_cast<half*>(buffer_cutlass_output.data_ptr), m, n, k, groupsize,
                         cutlass_gemm->getConfigs()[best_config_index], reinterpret_cast<char*>(buffer_ws.data_ptr),
                         ws_bytes, stream);
    }
    cudaEventRecord(begin, stream);
    for (size_t i = 0; i < 1000; ++i) {
      cutlass_gemm->gemm(reinterpret_cast<const half*>(buffer_input.data_ptr),
                         reinterpret_cast<const cutlass::uint4b_t*>(buffer_qweight.data_ptr),
                         reinterpret_cast<const half*>(buffer_weight_scales.data_ptr),
                         nullptr,  // no zeros
                         nullptr,  // no bias
                         reinterpret_cast<half*>(buffer_cutlass_output.data_ptr), m, n, k, groupsize,
                         cutlass_gemm->getConfigs()[best_config_index], reinterpret_cast<char*>(buffer_ws.data_ptr),
                         ws_bytes, stream);
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float cutlass_time;
    cudaEventElapsedTime(&cutlass_time, begin, end);

    int arch = llm_kernels::utils::GetSMVersion();
    if (!weight_only::is_supported(arch, weight_only::KernelType::FP16Int4Groupwise)) {
      throw std::runtime_error("Not support cuda kernel for type: FP16Int4Groupwise in current arch.");
    }
    weight_only::Params params{reinterpret_cast<const void*>(buffer_input.data_ptr),
                               nullptr,
                               reinterpret_cast<const void*>(buffer_qweight.data_ptr),
                               reinterpret_cast<const void*>(buffer_weight_scales.data_ptr),
                               nullptr,  // no zeros
                               nullptr,  // no bias
                               reinterpret_cast<void*>(buffer_cuda_output.data_ptr),
                               1.0f,
                               static_cast<int>(m),
                               static_cast<int>(n),
                               static_cast<int>(k),
                               static_cast<int>(groupsize),
                               weight_only::KernelType::FP16Int4Groupwise};
    for (size_t i = 0; i < 10; ++i) {
      weight_only::kernel_launcher(arch, params, stream);
    }
    cudaEventRecord(begin, stream);
    for (size_t i = 0; i < 1000; ++i) {
      weight_only::kernel_launcher(arch, params, stream);
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float cuda_time;
    cudaEventElapsedTime(&cuda_time, begin, end);

    cudaEventDestroy(begin);
    cudaEventDestroy(end);

    EXPECT_TRUE(cuda_time < cutlass_time);

    EXPECT_TRUE(CheckResult<half>("TestAsymmetricGemm_m" + std::to_string(m) + "_n" + std::to_string(n) + "_k" +
                                   std::to_string(k) + "_g" + std::to_string(groupsize),
                                   buffer_cuda_output, buffer_cutlass_output, 1e-1f, 1e-5f));

    DeleteBuffer(buffer_ws);
    DeleteBuffer(buffer_cutlass_output);
    DeleteBuffer(buffer_cuda_output);
    DeleteBuffer(buffer_input);
    DeleteBuffer(buffer_qweight);
    DeleteBuffer(buffer_weight_scales);
  }
};

TEST_F(NvidiaAsymmetricGemmTestSuit, AsymmetricGemmTest) {
  TestAsymmetricGemm(1, 5120, 5120, 128);
  TestAsymmetricGemm(1, 5120, 13824, 128);
  TestAsymmetricGemm(1, 13824, 5120, 128);
  TestAsymmetricGemm(1, 15360, 5120, 128);

  TestAsymmetricGemm(1, 5120, 2560, 128);
  TestAsymmetricGemm(1, 5120, 6912, 128);
  TestAsymmetricGemm(1, 6912, 5120, 128);
  TestAsymmetricGemm(1, 7680, 5120, 128);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels