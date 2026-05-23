/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include <torch/torch.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "dequant_fp4.h"

using namespace llm_kernels::nvidia;

namespace llm_kernels {
namespace nvidia {
namespace test {

class NvidiaDequantFp4TestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const size_t warmup = 100;
  const size_t iters = 1000;

  static float Fp4ToFloat(uint8_t nibble) {
    static const float lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    return lut[nibble & 0xF];
  }

  void RunPrecision(size_t num_experts, size_t n, size_t k) {
    const size_t total_elements = num_experts * n * k;
    const size_t num_groups_128 = total_elements / 128;

    torch::Tensor h_weight = torch::randint(0, 256, {static_cast<int64_t>(total_elements / 2)},
                                            torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
    torch::Tensor h_scale = torch::randint(119, 123, {static_cast<int64_t>(total_elements / 32)},
                                           torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));

    torch::Tensor d_weight = h_weight.cuda();
    torch::Tensor d_scale = h_scale.cuda();
    torch::Tensor d_output_fp8 = torch::empty({static_cast<int64_t>(total_elements)},
                                              torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat8_e4m3fn));
    torch::Tensor d_output_scale = torch::empty({static_cast<int64_t>(num_groups_128)},
                                                torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));

    dequant::dequant_fp4_fp8_launcher(stream, d_output_fp8.data_ptr(), d_output_scale.data_ptr(), d_weight.data_ptr(),
                                      d_scale.data_ptr(), num_groups_128);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto weight_ptr = h_weight.data_ptr<uint8_t>();
    auto scale_ptr = h_scale.data_ptr<uint8_t>();

    torch::Tensor gpu_fp8 = d_output_fp8.to(torch::kFloat32).cpu();
    torch::Tensor gpu_scale = d_output_scale.to(torch::kFloat32).cpu();
    auto gpu_fp8_ptr = gpu_fp8.data_ptr<float>();
    auto gpu_scale_ptr = gpu_scale.data_ptr<float>();

    size_t fp8_mismatch = 0;
    size_t scale_mismatch = 0;

    for (size_t g = 0; g < num_groups_128; ++g) {
      uint8_t e0 = scale_ptr[g * 4 + 0];
      uint8_t e1 = scale_ptr[g * 4 + 1];
      uint8_t e2 = scale_ptr[g * 4 + 2];
      uint8_t e3 = scale_ptr[g * 4 + 3];
      uint8_t max_exp = std::max({e0, e1, e2, e3});

      float expected_scale = exp2f(static_cast<float>(static_cast<int>(max_exp) - 127));
      if (std::abs(gpu_scale_ptr[g] - expected_scale) > 1e-6f) {
        ++scale_mismatch;
      }

      float rescale[4] = {exp2f(static_cast<float>(static_cast<int>(e0) - static_cast<int>(max_exp))),
                          exp2f(static_cast<float>(static_cast<int>(e1) - static_cast<int>(max_exp))),
                          exp2f(static_cast<float>(static_cast<int>(e2) - static_cast<int>(max_exp))),
                          exp2f(static_cast<float>(static_cast<int>(e3) - static_cast<int>(max_exp)))};

      for (size_t i = 0; i < 64; ++i) {
        uint8_t packed = weight_ptr[g * 64 + i];
        size_t k_pos_lo = i * 2, k_pos_hi = i * 2 + 1;
        float exp_lo = Fp4ToFloat(packed & 0xF) * rescale[k_pos_lo / 32];
        float exp_hi = Fp4ToFloat((packed >> 4) & 0xF) * rescale[k_pos_hi / 32];

        if (std::abs(gpu_fp8_ptr[g * 128 + k_pos_lo] - exp_lo) > 1e-6f) ++fp8_mismatch;
        if (std::abs(gpu_fp8_ptr[g * 128 + k_pos_hi] - exp_hi) > 1e-6f) ++fp8_mismatch;
      }
    }

    EXPECT_EQ(fp8_mismatch, 0);
    EXPECT_EQ(scale_mismatch, 0);
    printf("  Precision [E=%zu, N=%zu, K=%zu]: %zu fp8 + %zu scales OK\n", num_experts, n, k, total_elements,
           num_groups_128);
  }

  void RunEndToEnd(size_t num_experts, size_t n, size_t k) {
    const size_t total_elements = num_experts * n * k;
    const size_t num_groups_128 = total_elements / 128;

    torch::Tensor h_weight = torch::randint(0, 256, {static_cast<int64_t>(total_elements / 2)},
                                            torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
    torch::Tensor h_scale = torch::randint(118, 124, {static_cast<int64_t>(total_elements / 32)},
                                           torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));

    torch::Tensor d_weight = h_weight.cuda();
    torch::Tensor d_scale = h_scale.cuda();
    torch::Tensor d_output_fp8 = torch::empty({static_cast<int64_t>(total_elements)},
                                              torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat8_e4m3fn));
    torch::Tensor d_output_merged = torch::empty({static_cast<int64_t>(num_groups_128)},
                                                 torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));

    dequant::dequant_fp4_fp8_launcher(stream, d_output_fp8.data_ptr(), d_output_merged.data_ptr(), d_weight.data_ptr(),
                                      d_scale.data_ptr(), num_groups_128);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto weight_ptr = h_weight.data_ptr<uint8_t>();
    auto scale_ptr = h_scale.data_ptr<uint8_t>();

    torch::Tensor gpu_fp8 = d_output_fp8.to(torch::kFloat32).cpu();
    torch::Tensor gpu_merged = d_output_merged.to(torch::kFloat32).cpu();
    auto fp8_ptr = gpu_fp8.data_ptr<float>();
    auto merged_ptr = gpu_merged.data_ptr<float>();

    size_t mismatch = 0;
    for (size_t i = 0; i < total_elements / 2; ++i) {
      uint8_t packed = weight_ptr[i];
      size_t elem_lo = i * 2, elem_hi = i * 2 + 1;

      float scale_lo = exp2f(static_cast<float>(static_cast<int>(scale_ptr[elem_lo / 32]) - 127));
      float scale_hi = exp2f(static_cast<float>(static_cast<int>(scale_ptr[elem_hi / 32]) - 127));

      float true_lo = Fp4ToFloat(packed & 0xF) * scale_lo;
      float true_hi = Fp4ToFloat((packed >> 4) & 0xF) * scale_hi;

      float recon_lo = fp8_ptr[elem_lo] * merged_ptr[elem_lo / 128];
      float recon_hi = fp8_ptr[elem_hi] * merged_ptr[elem_hi / 128];

      if (std::abs(recon_lo - true_lo) > 1e-6f) ++mismatch;
      if (std::abs(recon_hi - true_hi) > 1e-6f) ++mismatch;
    }

    EXPECT_EQ(mismatch, 0);
    printf("  End-to-end [E=%zu, N=%zu, K=%zu]: mismatches=%zu/%zu\n", num_experts, n, k, mismatch, total_elements);
  }

  void RunPerformance(size_t num_experts, size_t n, size_t k) {
    const size_t total_elements = num_experts * n * k;
    const size_t num_groups_128 = total_elements / 128;

    BufferMeta input_weight = CreateBuffer<uint8_t>(MemoryType::MEMORY_GPU, {num_experts, n, k / 2}, true);
    BufferMeta input_scale = CreateBuffer<uint8_t>(MemoryType::MEMORY_GPU, {num_experts, n, k / 32}, true);
    BufferMeta output_fp8 = CreateBuffer<char>(MemoryType::MEMORY_GPU, {num_experts, n, k}, false);
    BufferMeta output_scale = CreateBuffer<uint16_t>(MemoryType::MEMORY_GPU, {num_experts, n, k / 128}, false);

    auto cuda_run = [&]() {
      dequant::dequant_fp4_fp8_launcher(stream, output_fp8.data_ptr, output_scale.data_ptr, input_weight.data_ptr,
                                        input_scale.data_ptr, num_groups_128);
    };
    float time = MeasureCudaExecutionTime(cuda_run, stream, warmup, iters);

    size_t total_bytes = total_elements / 2 + total_elements / 32 + total_elements + num_groups_128 * 2;
    double bandwidth_gb_s = static_cast<double>(total_bytes) / (time * 1e-3) / (1024.0 * 1024.0 * 1024.0);

    printf("  Performance [E=%zu, N=%zu, K=%zu]: %.4f ms, %.2f GB/s\n", num_experts, n, k, time, bandwidth_gb_s);
  }
};

TEST_F(NvidiaDequantFp4TestSuit, Fp4DequantPrecision) {
#if defined(ENABLE_FUSED_MOE_GPTQ_INT4_FP8)
  RunPrecision(48, 6144, 7168);
  RunPrecision(48, 7168, 3072);
#else
  std::cerr << "SM version is lower than 90. skipping dequant_fp4 kernel." << std::endl;
#endif
}

TEST_F(NvidiaDequantFp4TestSuit, Fp4DequantEndToEnd) {
#if defined(ENABLE_FUSED_MOE_GPTQ_INT4_FP8)
  RunEndToEnd(48, 6144, 7168);
  RunEndToEnd(48, 7168, 3072);
#else
  std::cerr << "SM version is lower than 90. skipping dequant_fp4 kernel." << std::endl;
#endif
}

TEST_F(NvidiaDequantFp4TestSuit, Fp4DequantPerformance) {
#if defined(ENABLE_FUSED_MOE_GPTQ_INT4_FP8)
  RunPerformance(48, 6144, 7168);
  RunPerformance(48, 7168, 3072);
#else
  std::cerr << "SM version is lower than 90. skipping dequant_fp4 kernel." << std::endl;
#endif
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
