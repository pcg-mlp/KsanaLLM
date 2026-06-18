/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <cmath>
#include <random>
#include <vector>

#include "cuda_bf16.h"
#include "cuda_runtime.h"
#include "test.h"

#include "device/nvidia/cuda_utils.h"
#include "kernels/nvidia/moe_kernel_wrapper.h"

namespace ksana_llm {

class MoeSumDispatchTest : public testing::Test {
 public:
  void SetUp() override {
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaStreamCreate(&stream));
  }
  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  int device{-1};
  cudaStream_t stream;
};

TEST_F(MoeSumDispatchTest, NonEP_BF16_DefaultPath) {
  const int num_tokens = 32, topk = 8, hidden_size = 2048;
  const int input_size = num_tokens * topk * hidden_size;
  const int output_size = num_tokens * hidden_size;

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<__nv_bfloat16> h_input(input_size);
  for (auto& v : h_input) v = __float2bfloat16(dist(gen));

  void *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  // scale=1.0 triggers InvokeMoeSum (original path, backwards compatible)
  InvokeMoeSumDispatch<__nv_bfloat16, false>(d_input, d_output, nullptr, num_tokens, topk, hidden_size, 1.0f, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> h_output(output_size);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  float max_err = 0;
  for (int t = 0; t < num_tokens; ++t) {
    for (int d = 0; d < hidden_size; ++d) {
      float acc = 0.f;
      for (int k = 0; k < topk; ++k) {
        acc += __bfloat162float(h_input[t * topk * hidden_size + k * hidden_size + d]);
      }
      float diff = std::abs(__bfloat162float(h_output[t * hidden_size + d]) - acc);
      max_err = std::max(max_err, diff);
    }
  }
  EXPECT_LT(max_err, 0.1f);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
}

TEST_F(MoeSumDispatchTest, NonEP_BF16_VecPath) {
  const int num_tokens = 32, topk = 8, hidden_size = 2048;
  const int input_size = num_tokens * topk * hidden_size;
  const int output_size = num_tokens * hidden_size;
  const float scale = 2.5f;

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<__nv_bfloat16> h_input(input_size);
  for (auto& v : h_input) v = __float2bfloat16(dist(gen));

  void *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  // scale!=1.0 triggers InvokeMoeSumVec (vectorized with fused scaling)
  InvokeMoeSumDispatch<__nv_bfloat16, false>(d_input, d_output, nullptr, num_tokens, topk, hidden_size, scale, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> h_output(output_size);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  float max_err = 0;
  for (int t = 0; t < num_tokens; ++t) {
    for (int d = 0; d < hidden_size; ++d) {
      float acc = 0.f;
      for (int k = 0; k < topk; ++k) {
        acc += __bfloat162float(h_input[t * topk * hidden_size + k * hidden_size + d]);
      }
      acc *= scale;
      float diff = std::abs(__bfloat162float(h_output[t * hidden_size + d]) - acc);
      max_err = std::max(max_err, diff);
    }
  }
  EXPECT_LT(max_err, 0.1f);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
}

TEST_F(MoeSumDispatchTest, EP_BF16_PartialExperts) {
  const int num_tokens = 32, topk = 8, hidden_size = 2048;
  const int input_size = num_tokens * topk * hidden_size;
  const int output_size = num_tokens * hidden_size;

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<__nv_bfloat16> h_input(input_size);
  for (auto& v : h_input) v = __float2bfloat16(dist(gen));

  // topk_ids: first 4 valid, last 4 invalid (-1)
  std::vector<int32_t> h_topk_ids(num_tokens * topk);
  for (int t = 0; t < num_tokens; ++t) {
    for (int k = 0; k < topk; ++k) {
      h_topk_ids[t * topk + k] = (k < 4) ? k : -1;
    }
  }

  void *d_input, *d_output;
  int32_t* d_topk_ids;
  CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_topk_ids), num_tokens * topk * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_topk_ids, h_topk_ids.data(), num_tokens * topk * sizeof(int32_t), cudaMemcpyHostToDevice));

  InvokeMoeSumDispatch<__nv_bfloat16, true>(d_input, d_output, d_topk_ids, num_tokens, topk, hidden_size, 1.0f, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> h_output(output_size);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  // CPU reference: sum only experts with non-negative topk_ids
  float max_err = 0;
  for (int t = 0; t < num_tokens; ++t) {
    for (int d = 0; d < hidden_size; ++d) {
      float acc = 0.f;
      for (int k = 0; k < topk; ++k) {
        if (h_topk_ids[t * topk + k] >= 0) {
          acc += __bfloat162float(h_input[t * topk * hidden_size + k * hidden_size + d]);
        }
      }
      float diff = std::abs(__bfloat162float(h_output[t * hidden_size + d]) - acc);
      max_err = std::max(max_err, diff);
    }
  }
  EXPECT_LT(max_err, 0.1f);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_topk_ids));
}

}  // namespace ksana_llm
