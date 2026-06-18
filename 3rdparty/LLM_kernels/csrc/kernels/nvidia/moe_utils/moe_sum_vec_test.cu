/* Copyright 2026 Tencent Inc.  All rights reserved.
 * Unit test: compare MoeSum (old) vs MoeSumVec (new vectorized)
 ==============================================================================*/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "csrc/kernels/nvidia/moe_utils/moe_sum_vec.h"
#include "csrc/kernels/nvidia/moe_utils/moe_utils.h"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class MoeSumVecTest : public testing::Test {
 protected:
  void SetUp() override { cudaStreamCreate(&stream_); }
  void TearDown() override {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
  cudaStream_t stream_;

  // CPU reference
  void CpuMoeSum(const std::vector<float>& input, std::vector<float>& output, int num_tokens, int topk,
                 int hidden_size, float scale) {
    output.resize(num_tokens * hidden_size, 0.f);
    for (int t = 0; t < num_tokens; ++t) {
      for (int d = 0; d < hidden_size; ++d) {
        float acc = 0.f;
        for (int k = 0; k < topk; ++k) {
          acc += input[t * topk * hidden_size + k * hidden_size + d];
        }
        output[t * hidden_size + d] = acc * scale;
      }
    }
  }

  template <typename T>
  void TestCorrectness(int num_tokens, int topk, int hidden_size, float scale) {
    std::mt19937 gen(42 + num_tokens);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    int input_size = num_tokens * topk * hidden_size;
    int output_size = num_tokens * hidden_size;

    std::vector<float> h_input_f(input_size);
    for (auto& v : h_input_f) v = dist(gen);

    // CPU reference (uses scale)
    std::vector<float> ref_output;
    CpuMoeSum(h_input_f, ref_output, num_tokens, topk, hidden_size, scale);

    // Convert to T
    std::vector<T> h_input(input_size);
    for (int i = 0; i < input_size; ++i) h_input[i] = static_cast<T>(h_input_f[i]);

    // GPU: old kernel (no scale built-in, so compare with scale=1 for old, then check new)
    void *d_input, *d_output_new;
    cudaMalloc(&d_input, input_size * sizeof(T));
    cudaMalloc(&d_output_new, output_size * sizeof(T));
    cudaMemcpy(d_input, h_input.data(), input_size * sizeof(T), cudaMemcpyHostToDevice);

    // Run NEW vectorized kernel
    InvokeMoeSumVec<T>(d_input, d_output_new, num_tokens, topk, hidden_size, scale, stream_);
    cudaStreamSynchronize(stream_);

    // Read back
    std::vector<T> h_output_new(output_size);
    cudaMemcpy(h_output_new.data(), d_output_new, output_size * sizeof(T), cudaMemcpyDeviceToHost);

    // Compare new vs CPU ref
    float max_err = 0;
    for (int i = 0; i < output_size; ++i) {
      float diff = std::abs(static_cast<float>(h_output_new[i]) - ref_output[i]);
      max_err = std::max(max_err, diff);
    }

    float atol = std::is_same_v<T, float> ? 1e-5f : (std::is_same_v<T, half> ? 5e-2f : 8e-2f);
    std::cout << "  tokens=" << num_tokens << " topk=" << topk << " hidden=" << hidden_size
              << " scale=" << scale << " max_err=" << max_err << " atol=" << atol << std::endl;
    EXPECT_LT(max_err, atol) << "MoeSumVec output error too large";

    cudaFree(d_input);
    cudaFree(d_output_new);
  }

  template <typename T>
  void BenchmarkKernels(int num_tokens, int topk, int hidden_size) {
    int input_size = num_tokens * topk * hidden_size;
    int output_size = num_tokens * hidden_size;

    std::vector<T> h_input(input_size, static_cast<T>(0.1f));
    void *d_input, *d_output_old, *d_output_new;
    int32_t* d_topk_ids;
    cudaMalloc(&d_input, input_size * sizeof(T));
    cudaMalloc(&d_output_old, output_size * sizeof(T));
    cudaMalloc(&d_output_new, output_size * sizeof(T));
    cudaMalloc(&d_topk_ids, num_tokens * topk * sizeof(int32_t));
    cudaMemcpy(d_input, h_input.data(), input_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(d_topk_ids, 0, num_tokens * topk * sizeof(int32_t));

    auto old_kernel = [&]() {
      InvokeMoeSum<T, false>(d_input, d_output_old, d_topk_ids, num_tokens, topk, hidden_size, stream_);
    };
    auto new_kernel = [&]() {
      InvokeMoeSumVec<T>(d_input, d_output_new, num_tokens, topk, hidden_size, 1.0f, stream_);
    };

    float old_ms = llm_kernels::utils::MeasureCudaExecutionTime(old_kernel, stream_, 10, 200);
    float new_ms = llm_kernels::utils::MeasureCudaExecutionTime(new_kernel, stream_, 10, 200);

    std::cout << "  tokens=" << num_tokens << " topk=" << topk << " hidden=" << hidden_size
              << " OLD=" << old_ms * 1000 << "us NEW=" << new_ms * 1000 << "us"
              << " speedup=" << old_ms / new_ms << "x" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output_old);
    cudaFree(d_output_new);
    cudaFree(d_topk_ids);
  }
};

// Correctness
TEST_F(MoeSumVecTest, BF16_Qwen3) { TestCorrectness<__nv_bfloat16>(32, 8, 2048, 1.0f); }
TEST_F(MoeSumVecTest, BF16_Decode) { TestCorrectness<__nv_bfloat16>(1, 8, 2048, 1.0f); }
TEST_F(MoeSumVecTest, BF16_Large) { TestCorrectness<__nv_bfloat16>(256, 8, 2048, 1.0f); }
TEST_F(MoeSumVecTest, BF16_Scaled) { TestCorrectness<__nv_bfloat16>(32, 8, 2048, 2.5f); }
TEST_F(MoeSumVecTest, FP16_Qwen3) { TestCorrectness<half>(32, 8, 2048, 1.0f); }
TEST_F(MoeSumVecTest, Float_Qwen3) { TestCorrectness<float>(32, 8, 2048, 1.0f); }

// EP path: InvokeMoeSum<T, true> with some topk_ids negative (skipped experts)
TEST_F(MoeSumVecTest, BF16_EP_PartialExperts) {
  const int num_tokens = 32, topk = 8, hidden_size = 2048;
  const int input_size = num_tokens * topk * hidden_size;
  const int output_size = num_tokens * hidden_size;

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> h_input_f(input_size);
  for (auto& v : h_input_f) v = dist(gen);

  // Build topk_ids: mark some experts as invalid (-1)
  std::vector<int32_t> h_topk_ids(num_tokens * topk);
  for (int t = 0; t < num_tokens; ++t) {
    for (int k = 0; k < topk; ++k) {
      h_topk_ids[t * topk + k] = (k < 4) ? k : -1;
    }
  }

  // CPU reference: sum only experts with non-negative topk_ids
  std::vector<float> ref_output(output_size, 0.f);
  for (int t = 0; t < num_tokens; ++t) {
    for (int d = 0; d < hidden_size; ++d) {
      float acc = 0.f;
      for (int k = 0; k < topk; ++k) {
        if (h_topk_ids[t * topk + k] >= 0) {
          acc += h_input_f[t * topk * hidden_size + k * hidden_size + d];
        }
      }
      ref_output[t * hidden_size + d] = acc;
    }
  }

  std::vector<__nv_bfloat16> h_input(input_size);
  for (int i = 0; i < input_size; ++i) h_input[i] = __float2bfloat16(h_input_f[i]);

  void *d_input, *d_output;
  int32_t* d_topk_ids;
  cudaMalloc(&d_input, input_size * sizeof(__nv_bfloat16));
  cudaMalloc(&d_output, output_size * sizeof(__nv_bfloat16));
  cudaMalloc(&d_topk_ids, num_tokens * topk * sizeof(int32_t));
  cudaMemcpy(d_input, h_input.data(), input_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_topk_ids, h_topk_ids.data(), num_tokens * topk * sizeof(int32_t), cudaMemcpyHostToDevice);

  InvokeMoeSum<__nv_bfloat16, true>(d_input, d_output, d_topk_ids, num_tokens, topk, hidden_size, stream_);
  cudaStreamSynchronize(stream_);

  std::vector<__nv_bfloat16> h_output(output_size);
  cudaMemcpy(h_output.data(), d_output, output_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

  float max_err = 0;
  for (int i = 0; i < output_size; ++i) {
    float diff = std::abs(__bfloat162float(h_output[i]) - ref_output[i]);
    max_err = std::max(max_err, diff);
  }
  std::cout << "  EP partial experts: max_err=" << max_err << std::endl;
  EXPECT_LT(max_err, 8e-2f);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_topk_ids);
}

// Performance
TEST_F(MoeSumVecTest, Perf_BF16_Decode) { BenchmarkKernels<__nv_bfloat16>(1, 8, 2048); }
TEST_F(MoeSumVecTest, Perf_BF16_Small) { BenchmarkKernels<__nv_bfloat16>(32, 8, 2048); }
TEST_F(MoeSumVecTest, Perf_BF16_Large) { BenchmarkKernels<__nv_bfloat16>(256, 8, 2048); }

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
