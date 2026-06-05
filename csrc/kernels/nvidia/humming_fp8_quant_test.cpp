/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <iostream>

#include "cuda_fp8.h"
#include "cuda_runtime.h"
#include "test.h"

#include "csrc/kernels/nvidia/moe_utils/moe_utils.h"
#include "device/device_types.h"
#include "device/nvidia/cuda_utils.h"
#include "kernels/nvidia/kernel_wrapper.h"
#include "kernels/nvidia/triton_wrapper.h"
#include "utils/singleton.h"

namespace ksana_llm {

class HummingFp8QuantTestSuit : public testing::Test {
 public:
  void SetUp() override {
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

 protected:
  int32_t rank{0};
  int32_t device{-1};
  cudaStream_t stream;

  bool ValidTestHardware() {
    int major = 0, minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    return major >= 9;
  }

  static constexpr int kWarmupIters = 10;
  static constexpr int kBenchIters = 100;

  void RunPrecisionTest(int hidden_size, int num_tokens) {
    const auto dtype = GetTorchDataType<__nv_bfloat16>();
    torch::Tensor d_input =
        torch::randn({num_tokens, hidden_size}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(dtype));

    auto fp8_opt = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat8_e4m3fn);
    torch::Tensor d_triton_out = torch::empty({num_tokens, hidden_size}, fp8_opt);
    torch::Tensor d_triton_scale =
        torch::empty({num_tokens}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32));

    torch::Tensor d_cuda_out = torch::empty_like(d_triton_out);
    torch::Tensor d_cuda_scale = torch::empty_like(d_triton_scale);

    Singleton<TritonWrapper>::GetInstance()->InvokeHummingFp8Quant<__nv_bfloat16>(
        static_cast<const __nv_bfloat16*>(d_input.data_ptr()), static_cast<float*>(d_triton_scale.data_ptr()),
        d_triton_out.data_ptr(), hidden_size, num_tokens, stream);

    llm_kernels::nvidia::InvokeDynamicPerTokenScaledFP8Quant<__nv_bfloat16>(
        static_cast<const __nv_bfloat16*>(d_input.data_ptr()), static_cast<const float*>(d_cuda_scale.data_ptr()),
        nullptr, d_cuda_out.data_ptr(), hidden_size, num_tokens, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t fp8_mismatches = torch::sum(d_triton_out.to(torch::kInt32) != d_cuda_out.to(torch::kInt32)).item<int64_t>();
    float fp8_mismatch_rate = static_cast<float>(fp8_mismatches) / static_cast<float>(num_tokens * hidden_size);
    EXPECT_LT(fp8_mismatch_rate, 0.01f) << "FP8 output mismatch rate " << fp8_mismatch_rate
                                        << " exceeds 1% for hidden_size=" << hidden_size << " tokens=" << num_tokens;

    float max_scale_rel_err =
        (torch::abs(d_triton_scale - d_cuda_scale) / torch::clamp(torch::abs(d_cuda_scale), 1e-10f, INFINITY))
            .max()
            .item<float>();
    EXPECT_LT(max_scale_rel_err, 1e-4f) << "Scale relative error " << max_scale_rel_err
                                        << " exceeds 1e-4 for hidden_size=" << hidden_size << " tokens=" << num_tokens;

    std::cout << "hidden_size=" << hidden_size << " tokens=" << num_tokens << ": FP8 mismatches=" << fp8_mismatches
              << " (rate=" << fp8_mismatch_rate << "), max scale rel err=" << max_scale_rel_err << std::endl;
  }

  void RunPerfTest(int hidden_size, int num_tokens) {
    const auto dtype = GetTorchDataType<__nv_bfloat16>();
    torch::Tensor d_input =
        torch::randn({num_tokens, hidden_size}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(dtype));

    auto fp8_opt = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat8_e4m3fn);
    torch::Tensor d_triton_out = torch::empty({num_tokens, hidden_size}, fp8_opt);
    torch::Tensor d_triton_scale =
        torch::empty({num_tokens}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32));
    torch::Tensor d_cuda_out = torch::empty_like(d_triton_out);
    torch::Tensor d_cuda_scale = torch::empty_like(d_triton_scale);

    auto triton_kernel = [&]() {
      Singleton<TritonWrapper>::GetInstance()->InvokeHummingFp8Quant<__nv_bfloat16>(
          static_cast<const __nv_bfloat16*>(d_input.data_ptr()), static_cast<float*>(d_triton_scale.data_ptr()),
          d_triton_out.data_ptr(), hidden_size, num_tokens, stream);
    };
    auto cuda_kernel = [&]() {
      llm_kernels::nvidia::InvokeDynamicPerTokenScaledFP8Quant<__nv_bfloat16>(
          static_cast<const __nv_bfloat16*>(d_input.data_ptr()), static_cast<const float*>(d_cuda_scale.data_ptr()),
          nullptr, d_cuda_out.data_ptr(), hidden_size, num_tokens, stream);
    };

    float triton_avg_ms = MeasureCudaExecutionTime(triton_kernel, stream, kWarmupIters, kBenchIters, /*flush_l2=*/true);
    float cuda_avg_ms = MeasureCudaExecutionTime(cuda_kernel, stream, kWarmupIters, kBenchIters, /*flush_l2=*/true);

    // Bandwidth: input (bf16) + output (fp8) + scale (fp32 per token) read/written
    size_t bytes = static_cast<size_t>(num_tokens) * hidden_size * sizeof(__nv_bfloat16)  // input
                   + static_cast<size_t>(num_tokens) * hidden_size * sizeof(uint8_t)      // fp8 output
                   + static_cast<size_t>(num_tokens) * sizeof(float);                     // scale

    float triton_bw = bytes / (triton_avg_ms * 1e6f);  // GB/s
    float cuda_bw = bytes / (cuda_avg_ms * 1e6f);      // GB/s

    std::cout << "hidden_size=" << hidden_size << " tokens=" << num_tokens << ": Triton=" << triton_avg_ms << " ms ("
              << triton_bw << " GB/s)"
              << ", CUDA=" << cuda_avg_ms << " ms (" << cuda_bw << " GB/s)"
              << ", speedup=" << cuda_avg_ms / triton_avg_ms << "x" << std::endl;
  }
};

TEST_F(HummingFp8QuantTestSuit, HummingFp8QuantPrecisionTest) {
  if (!ValidTestHardware()) {
    std::cout << "Skipping test because SM version is less than 90" << std::endl;
    return;
  }

  // All deduplicated N values from Flash (H=4096, I=2048) and Pro (H=7168, I=3072)
  // with TP in {1,2,4,8} and EP in {1,2,4,8}.  moe_tp = max(1, tp/ep); N1=H; N2=ceil(I/moe_tp).
  constexpr int kCommonN[] = {256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 7168};
  for (int hidden_size : kCommonN) {
    for (int num_tokens : {1, 50, 100, 4096, 8192}) {
      RunPrecisionTest(hidden_size, num_tokens);
    }
  }
}

TEST_F(HummingFp8QuantTestSuit, HummingFp8QuantPerfTest) {
  if (!ValidTestHardware()) {
    std::cout << "Skipping test because SM version is less than 90" << std::endl;
    return;
  }

  const std::vector<int> kPerfN = {256, 512, 1024, 2048, 4096, 7168};
  const std::vector<int> kPerfTokens = {1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

  std::cout << "\n========== Humming FP8 Quant Performance (bf16) ==========" << std::endl;
  for (int hidden_size : kPerfN) {
    for (int num_tokens : kPerfTokens) {
      RunPerfTest(hidden_size, num_tokens);
    }
    std::cout << "---------------------------------------------------------" << std::endl;
  }
}

}  // namespace ksana_llm