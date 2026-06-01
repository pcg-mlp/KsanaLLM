/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <cstring>
#include <vector>

#include "cuda_bf16.h"
#include "cuda_runtime.h"
#include "test.h"

#include "device/nvidia/cuda_utils.h"
#include "kernels/nvidia/moe_kernel_wrapper.h"
#include "kernels/nvidia/triton_wrapper.h"
#include "utils/singleton.h"

namespace ksana_llm {

class FusedBatchedMoeTest : public testing::Test {
 public:
  void SetUp() override {
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaStreamCreate(&stream));

    int major = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    if (major < 9) {
      GTEST_SKIP() << "Skipping: requires SM >= 90";
    }
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (stream) {
      CUDA_CHECK(cudaStreamDestroy(stream));
    }
  }

 protected:
  int device{-1};
  cudaStream_t stream{nullptr};
};

// 零输入 smoke test: 全零 input/weight → 输出应全零
TEST_F(FusedBatchedMoeTest, ZeroInputProducesZeroOutput) {
  const char* skip_env = std::getenv("SKIP_TEST_IF_INVALID_TRITON_VERSION");
  if (skip_env != nullptr && std::string(skip_env) == "1") {
    GTEST_SKIP() << "Skipping test due to Triton version incompatibility (SKIP_TEST_IF_INVALID_TRITON_VERSION=1).";
  }
  constexpr int num_experts = 8;
  constexpr int max_num_tokens = 4;
  constexpr int hidden_size = 128;
  constexpr int inter_size = 128;
  constexpr int topk = 8;

  const size_t input_bytes = num_experts * max_num_tokens * hidden_size * sizeof(__nv_bfloat16);
  const size_t w1_bytes = num_experts * (inter_size * 2) * hidden_size * sizeof(__nv_bfloat16);
  const size_t w2_bytes = num_experts * hidden_size * inter_size * sizeof(__nv_bfloat16);
  const size_t output_bytes = input_bytes;
  const size_t intermediate_bytes = num_experts * max_num_tokens * (inter_size * 2) * sizeof(__nv_bfloat16);
  const size_t silu_bytes = num_experts * max_num_tokens * inter_size * sizeof(__nv_bfloat16);
  const size_t expert_num_tokens_bytes = num_experts * sizeof(int);

  void *d_input, *d_w1, *d_w2, *d_output, *d_intermediate, *d_silu, *d_expert_num_tokens;
  CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
  CUDA_CHECK(cudaMalloc(&d_w1, w1_bytes));
  CUDA_CHECK(cudaMalloc(&d_w2, w2_bytes));
  CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
  CUDA_CHECK(cudaMalloc(&d_intermediate, intermediate_bytes));
  CUDA_CHECK(cudaMalloc(&d_silu, silu_bytes));
  CUDA_CHECK(cudaMalloc(&d_expert_num_tokens, expert_num_tokens_bytes));

  // 全零初始化
  CUDA_CHECK(cudaMemset(d_input, 0, input_bytes));
  CUDA_CHECK(cudaMemset(d_w1, 0, w1_bytes));
  CUDA_CHECK(cudaMemset(d_w2, 0, w2_bytes));
  CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));
  CUDA_CHECK(cudaMemset(d_intermediate, 0, intermediate_bytes));
  CUDA_CHECK(cudaMemset(d_silu, 0, silu_bytes));

  // 每个 expert 分配 max_num_tokens 个 token
  std::vector<int> h_expert_num_tokens(num_experts, max_num_tokens);
  CUDA_CHECK(
      cudaMemcpy(d_expert_num_tokens, h_expert_num_tokens.data(), expert_num_tokens_bytes, cudaMemcpyHostToDevice));

  InvokeFusedBatchedMoe<__nv_bfloat16>(d_output, d_input, d_w1, d_w2,
                                       /*input_scale=*/nullptr, /*weight1_scale=*/nullptr, /*weight2_scale=*/nullptr,
                                       num_experts, d_expert_num_tokens, max_num_tokens, hidden_size, inter_size, topk,
                                       /*use_fp8=*/false, /*use_int8_w8a8=*/false, /*use_int4=*/false,
                                       /*group_size=*/0, d_intermediate, d_silu, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // 验证输出全零
  std::vector<__nv_bfloat16> h_output(num_experts * max_num_tokens * hidden_size);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < h_output.size(); ++i) {
    EXPECT_EQ(__bfloat162float(h_output[i]), 0.0f) << "Non-zero at index " << i;
  }

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_w1));
  CUDA_CHECK(cudaFree(d_w2));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_intermediate));
  CUDA_CHECK(cudaFree(d_silu));
  CUDA_CHECK(cudaFree(d_expert_num_tokens));
}

}  // namespace ksana_llm
