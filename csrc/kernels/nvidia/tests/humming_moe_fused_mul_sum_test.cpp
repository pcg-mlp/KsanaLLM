/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <iostream>
#include <vector>

#include "cuda_runtime.h"

#include "csrc/kernels/nvidia/moe_utils/moe_utils.h"
#include "device/nvidia/cuda_utils.h"
#include "kernels/nvidia/kernel_wrapper.h"
#include "kernels/nvidia/triton_wrapper.h"
#include "test.h"
#include "utils/singleton.h"

namespace ksana_llm {

class HummingMoeFusedMulSumTestSuit : public testing::Test {
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

  using T = __nv_bfloat16;

  void RunPrecisionTest(int hidden_size, int num_tokens, int topk, bool is_ep) {
    int64_t expanded_tokens = static_cast<int64_t>(num_tokens) * topk;

    auto dtype = torch::kBFloat16;
    torch::Tensor d_input =
        torch::randn({expanded_tokens, hidden_size}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(dtype));
    torch::Tensor d_weights =
        torch::randn({expanded_tokens}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32));
    torch::Tensor d_ids = torch::randint(0, 256, {num_tokens, topk},
                                         torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt32));

    if (is_ep) {
      auto mask =
          torch::rand({num_tokens, topk}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32))
              .lt(1.0f / 8.0f);
      d_ids.masked_fill_(mask, -1);
    }

    torch::Tensor d_triton_out =
        torch::empty({num_tokens, hidden_size}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(dtype));
    torch::Tensor d_cuda_out = torch::empty_like(d_triton_out);

    Singleton<TritonWrapper>::GetInstance()->InvokeHummingMoeFusedMulSum<T>(
        d_input.data_ptr(), static_cast<const float*>(d_weights.data_ptr()), d_ids.data_ptr(), d_triton_out.data_ptr(),
        num_tokens, topk, hidden_size, is_ep, stream);

    if (is_ep) {
      llm_kernels::nvidia::InvokeMoeFusedMulSum<T, true>(
          d_input.data_ptr(), static_cast<const float*>(d_weights.data_ptr()), d_ids.data_ptr(), d_cuda_out.data_ptr(),
          num_tokens, topk, hidden_size, stream);
    } else {
      llm_kernels::nvidia::InvokeMoeFusedMulSum<T, false>(
          d_input.data_ptr(), static_cast<const float*>(d_weights.data_ptr()), d_ids.data_ptr(), d_cuda_out.data_ptr(),
          num_tokens, topk, hidden_size, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto diff = (d_triton_out.to(torch::kFloat32) - d_cuda_out.to(torch::kFloat32)).abs();
    auto rel_diff = diff / d_cuda_out.to(torch::kFloat32).abs().clamp(1e-6f, INFINITY);
    float max_rel_err = rel_diff.max().item<float>();
    float max_abs_err = diff.max().item<float>();

    EXPECT_LT(max_rel_err, 1e-2f) << "Max relative error " << max_rel_err << " exceeds 1% for "
                                  << "hidden_size=" << hidden_size << " tokens=" << num_tokens << " topk=" << topk
                                  << " is_ep=" << is_ep;
    EXPECT_LT(max_abs_err, 1e-3f) << "Max absolute error " << max_abs_err << " for hidden_size=" << hidden_size
                                  << " tokens=" << num_tokens << " topk=" << topk << " is_ep=" << is_ep;
  }

  void RunPerfTest(int hidden_size, int num_tokens, int topk, bool is_ep) {
    int64_t expanded_tokens = static_cast<int64_t>(num_tokens) * topk;

    auto dtype = torch::kBFloat16;
    torch::Tensor d_input =
        torch::randn({expanded_tokens, hidden_size}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(dtype));
    torch::Tensor d_weights =
        torch::randn({expanded_tokens}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32));
    torch::Tensor d_ids = torch::randint(0, 256, {num_tokens, topk},
                                         torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt32));

    if (is_ep) {
      auto mask =
          torch::rand({num_tokens, topk}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32))
              .lt(1.0f / 8.0f);
      d_ids.masked_fill_(mask, -1);
    }

    torch::Tensor d_triton_out =
        torch::empty({num_tokens, hidden_size}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(dtype));
    torch::Tensor d_cuda_out = torch::empty_like(d_triton_out);

    auto triton_kernel = [&]() {
      Singleton<TritonWrapper>::GetInstance()->InvokeHummingMoeFusedMulSum<T>(
          d_input.data_ptr(), static_cast<const float*>(d_weights.data_ptr()), d_ids.data_ptr(),
          d_triton_out.data_ptr(), num_tokens, topk, hidden_size, is_ep, stream);
    };

    auto cuda_kernel = [&]() {
      if (is_ep) {
        llm_kernels::nvidia::InvokeMoeFusedMulSum<T, true>(
            d_input.data_ptr(), static_cast<const float*>(d_weights.data_ptr()), d_ids.data_ptr(),
            d_cuda_out.data_ptr(), num_tokens, topk, hidden_size, stream);
      } else {
        llm_kernels::nvidia::InvokeMoeFusedMulSum<T, false>(
            d_input.data_ptr(), static_cast<const float*>(d_weights.data_ptr()), d_ids.data_ptr(),
            d_cuda_out.data_ptr(), num_tokens, topk, hidden_size, stream);
      }
    };

    float triton_ms = MeasureCudaExecutionTime(triton_kernel, stream, kWarmupIters, kBenchIters, /*flush_l2=*/true);
    float cuda_ms = MeasureCudaExecutionTime(cuda_kernel, stream, kWarmupIters, kBenchIters, /*flush_l2=*/true);

    // Total memory traffic: input(bf16,R) + weights(fp32,R) + ids(int32,R, EP only) + output(bf16,W)
    int64_t bytes = 2LL * num_tokens * topk * hidden_size      // input read
                    + 4LL * num_tokens * topk                  // weights read
                    + (is_ep ? 4LL * num_tokens * topk : 0LL)  // ids read (EP only)
                    + 2LL * num_tokens * hidden_size;          // output write
    float triton_gbps = static_cast<float>(bytes) / triton_ms / 1e6f;
    float cuda_gbps = static_cast<float>(bytes) / cuda_ms / 1e6f;

    std::cout << "Perf: H=" << hidden_size << " M=" << num_tokens << " topk=" << topk << " is_ep=" << is_ep
              << " Triton=" << triton_ms << " ms (" << triton_gbps << " GB/s)"
              << " CUDA=" << cuda_ms << " ms (" << cuda_gbps << " GB/s)"
              << " ratio=" << cuda_ms / triton_ms << "x" << std::endl;
  }
};

TEST_F(HummingMoeFusedMulSumTestSuit, HummingMoeFusedMulSumPrecisionTest) {
  if (!ValidTestHardware()) {
    std::cout << "Skipping test because SM version is less than 90" << std::endl;
    return;
  }

  // Test both EP modes across sizes matching Flash (4096) and Pro (7168).
  int topk = 6;
  for (int is_ep : {0, 1}) {
    for (int hidden_size : {4096, 7168}) {
      for (int num_tokens : {1, 4, 50, 128, 256, 1024, 4096}) {
        RunPrecisionTest(hidden_size, num_tokens, topk, static_cast<bool>(is_ep));
      }
    }
  }
}

TEST_F(HummingMoeFusedMulSumTestSuit, HummingMoeFusedMulSumPerfTest) {
  if (!ValidTestHardware()) {
    std::cout << "Skipping test because SM version is less than 90" << std::endl;
    return;
  }

  int topk = 6;
  std::cout << "\n========== MoE Fused Mul Sum Performance (bf16) ==========" << std::endl;
  for (int is_ep : {0, 1}) {
    for (int hidden_size : {4096, 7168}) {
      for (int num_tokens : {1, 4, 16, 64, 128, 256, 512, 1024, 4096, 8192}) {
        RunPerfTest(hidden_size, num_tokens, topk, static_cast<bool>(is_ep));
      }
      std::cout << "---------------------------------------------------------" << std::endl;
    }
  }
}

}  // namespace ksana_llm
