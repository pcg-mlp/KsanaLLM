/* Copyright 2026 Tencent Inc.  All rights reserved.
 * Unit test: FusedQKNormRoPE vs KsanaLLM's separate QKNorm + RoPE
 ==============================================================================*/

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include "csrc/kernels/nvidia/qknorm_rope/fused_qknorm_rope.h"
#include "csrc/kernels/nvidia/layernorm/layernorm.h"
#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::nvidia;

class FusedQKNormRoPETest : public testing::Test {
 protected:
  void SetUp() override { cudaStreamCreate(&stream_); }
  void TearDown() override {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
  cudaStream_t stream_;

  void TestCorrectness(int num_tokens, int num_heads, int head_dim) {
    const float eps = 1e-6f;
    const int total_size = num_tokens * num_heads * head_dim;
    const int max_pos = 4096;

    // Random data
    std::mt19937 gen(42 + num_tokens);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<__nv_bfloat16> h_input(total_size);
    for (auto& v : h_input) v = __float2bfloat16(dist(gen));
    std::vector<__nv_bfloat16> h_weight(head_dim);
    for (auto& v : h_weight) v = __float2bfloat16(dist(gen));
    std::vector<int64_t> h_pos(num_tokens);
    for (int i = 0; i < num_tokens; i++) h_pos[i] = i;
    std::vector<int64_t> h_mask(num_tokens, 1);

    // GPU buffers
    void *d_input, *d_output_fused, *d_output_ref, *d_weight, *d_cache, *d_pos, *d_mask;
    cudaMalloc(&d_input, total_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output_fused, total_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output_ref, total_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_weight, head_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_cache, max_pos * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_pos, num_tokens * sizeof(int64_t));
    cudaMalloc(&d_mask, num_tokens * sizeof(int64_t));

    cudaMemcpy(d_input, h_input.data(), total_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), head_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, h_pos.data(), num_tokens * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), num_tokens * sizeof(int64_t), cudaMemcpyHostToDevice);

    // ====== Setup RotaryEmbedding — this COMPUTES the cos_sin_cache ======
    RotaryEmbeddingCuda rope;
    rope.SetConfig<__nv_bfloat16>(d_cache, head_dim, max_pos, 10000.0f,
                                   head_dim, head_dim,
                                   num_heads, 0,  // num_kv_heads=0
                                   num_heads * head_dim,  // stride
                                   true, stream_);  // is_neox
    cudaStreamSynchronize(stream_);

    // ====== Run NEW fused kernel (using the SAME computed cache) ======
    InvokeFusedQKNormRoPE(d_input, d_output_fused, d_weight, d_pos, d_cache,
                          num_tokens, num_heads, head_dim, eps, stream_);

    // ====== Run OLD separate kernels (reference) ======
    // Copy input to ref output (QKNorm is in-place)
    cudaMemcpy(d_output_ref, d_input, total_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

    // QKNorm: treats input as [num_tokens, num_heads * head_dim]
    InvokeFusedQKVRmsNorm<__nv_bfloat16>(
        static_cast<__nv_bfloat16*>(d_output_ref),
        static_cast<const __nv_bfloat16*>(d_output_ref),
        static_cast<const __nv_bfloat16*>(d_weight),
        static_cast<const __nv_bfloat16*>(d_weight),  // same weight for K (not used)
        eps, num_tokens, num_heads, 0,  // num_kv_heads=0 to only process Q
        head_dim, static_cast<const int64_t*>(d_mask), stream_);

    // RoPE on the normed output
    rope.SetInput(static_cast<int64_t*>(d_pos), static_cast<int64_t*>(d_mask),
                  d_output_ref, nullptr, num_tokens, stream_);
    rope.Forward<__nv_bfloat16>();

    cudaStreamSynchronize(stream_);

    // Compare
    std::vector<__nv_bfloat16> h_fused(total_size), h_ref(total_size);
    cudaMemcpy(h_fused.data(), d_output_fused, total_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), d_output_ref, total_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float max_err = 0;
    int mismatch_count = 0;
    for (int i = 0; i < total_size; i++) {
      float a = __bfloat162float(h_fused[i]);
      float b = __bfloat162float(h_ref[i]);
      float err = std::abs(a - b);
      max_err = std::max(max_err, err);
      if (err > 0.05f) mismatch_count++;
    }

    std::cout << "  tokens=" << num_tokens << " heads=" << num_heads
              << " max_err=" << max_err << " mismatches(>0.05)=" << mismatch_count << std::endl;
    EXPECT_LT(max_err, 0.05f) << "Fused vs reference output mismatch too large";
    EXPECT_LT(mismatch_count, total_size / 1000) << "Too many elements with >0.05 error";

    cudaFree(d_input);
    cudaFree(d_output_fused);
    cudaFree(d_output_ref);
    cudaFree(d_weight);
    cudaFree(d_cache);
    cudaFree(d_pos);
    cudaFree(d_mask);
  }

  void BenchmarkBoth(int num_tokens, int num_heads, int head_dim) {
    const float eps = 1e-6f;
    const int total_size = num_tokens * num_heads * head_dim;
    const int max_pos = 4096;

    std::vector<__nv_bfloat16> h_input(total_size, __float2bfloat16(0.5f));
    std::vector<__nv_bfloat16> h_weight(head_dim, __float2bfloat16(1.0f));
    std::vector<int64_t> h_pos(num_tokens), h_mask(num_tokens, 1);
    for (int i = 0; i < num_tokens; i++) h_pos[i] = i;

    void *d_input, *d_output, *d_weight, *d_cache, *d_pos, *d_mask;
    cudaMalloc(&d_input, total_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output, total_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_weight, head_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_cache, max_pos * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_pos, num_tokens * sizeof(int64_t));
    cudaMalloc(&d_mask, num_tokens * sizeof(int64_t));
    cudaMemcpy(d_input, h_input.data(), total_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), head_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, h_pos.data(), num_tokens * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), num_tokens * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Setup RoPE (this computes the cache)
    RotaryEmbeddingCuda rope;
    rope.SetConfig<__nv_bfloat16>(d_cache, head_dim, max_pos, 10000.0f,
                                   head_dim, head_dim, num_heads, 0,
                                   num_heads * head_dim, true, stream_);
    cudaStreamSynchronize(stream_);

    auto new_kernel = [&]() {
      InvokeFusedQKNormRoPE(d_input, d_output, d_weight, d_pos, d_cache,
                            num_tokens, num_heads, head_dim, eps, stream_);
    };
    auto old_kernel = [&]() {
      InvokeFusedQKVRmsNorm<__nv_bfloat16>(
          static_cast<__nv_bfloat16*>(d_output), static_cast<const __nv_bfloat16*>(d_input),
          static_cast<const __nv_bfloat16*>(d_weight), static_cast<const __nv_bfloat16*>(d_weight),
          eps, num_tokens, num_heads, 0, head_dim, static_cast<const int64_t*>(d_mask), stream_);
      rope.SetInput(static_cast<int64_t*>(d_pos), static_cast<int64_t*>(d_mask), d_output, nullptr, num_tokens, stream_);
      rope.Forward<__nv_bfloat16>();
    };

    float new_ms = llm_kernels::utils::MeasureCudaExecutionTime(new_kernel, stream_, 10, 200);
    float old_ms = llm_kernels::utils::MeasureCudaExecutionTime(old_kernel, stream_, 10, 200);

    std::cout << "  tokens=" << num_tokens << " heads=" << num_heads
              << " OLD=" << old_ms * 1000 << "us"
              << " NEW=" << new_ms * 1000 << "us"
              << " speedup=" << old_ms / new_ms << "x" << std::endl;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_cache);
    cudaFree(d_pos);
    cudaFree(d_mask);
  }
};

// Correctness
TEST_F(FusedQKNormRoPETest, BF16_Decode_Q) { TestCorrectness(1, 24, 128); }
TEST_F(FusedQKNormRoPETest, BF16_Small_Q) { TestCorrectness(32, 24, 128); }
TEST_F(FusedQKNormRoPETest, BF16_Prefill_Q) { TestCorrectness(2048, 24, 128); }
TEST_F(FusedQKNormRoPETest, BF16_Decode_K) { TestCorrectness(1, 8, 128); }
TEST_F(FusedQKNormRoPETest, BF16_Prefill_K) { TestCorrectness(2048, 8, 128); }

// Mask: prefix caching scenario (first N tokens masked, rest active)
TEST_F(FusedQKNormRoPETest, Masked_PrefixCaching) {
  const int total_tokens = 1000, num_active = 200, num_heads = 24, head_dim = 128;
  const float eps = 1e-6f;
  const int stride_size = num_heads * head_dim;
  const int total_size = total_tokens * stride_size;
  const int max_pos = 4096;

  std::mt19937 gen(99);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<__nv_bfloat16> h_input(total_size);
  for (auto& v : h_input) v = __float2bfloat16(dist(gen));
  std::vector<__nv_bfloat16> h_weight(head_dim);
  for (auto& v : h_weight) v = __float2bfloat16(dist(gen));
  std::vector<int64_t> h_pos(total_tokens);
  for (int i = 0; i < total_tokens; i++) h_pos[i] = i;
  std::vector<int64_t> h_mask(total_tokens, 0);
  for (int i = total_tokens - num_active; i < total_tokens; i++) h_mask[i] = 1;
  std::vector<__nv_bfloat16> h_cache(max_pos * head_dim);
  for (auto& v : h_cache) v = __float2bfloat16(dist(gen));

  void *d_input, *d_out_masked, *d_out_ref, *d_weight, *d_cache, *d_pos, *d_mask;
  cudaMalloc(&d_input, total_size * sizeof(__nv_bfloat16));
  cudaMalloc(&d_out_masked, total_size * sizeof(__nv_bfloat16));
  cudaMalloc(&d_out_ref, total_size * sizeof(__nv_bfloat16));
  cudaMalloc(&d_weight, head_dim * sizeof(__nv_bfloat16));
  cudaMalloc(&d_cache, max_pos * head_dim * sizeof(__nv_bfloat16));
  cudaMalloc(&d_pos, total_tokens * sizeof(int64_t));
  cudaMalloc(&d_mask, total_tokens * sizeof(int64_t));
  cudaMemcpy(d_input, h_input.data(), total_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight.data(), head_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cache, h_cache.data(), max_pos * head_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos, h_pos.data(), total_tokens * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask.data(), total_tokens * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemset(d_out_masked, 0xAB, total_size * sizeof(__nv_bfloat16));

  InvokeFusedQKNormRoPE(d_input, d_out_masked, d_weight, d_pos, d_cache,
                         total_tokens, num_heads, head_dim, eps, stream_, stride_size,
                         static_cast<int64_t*>(d_mask));

  int offset_bytes = (total_tokens - num_active) * stride_size * sizeof(__nv_bfloat16);
  InvokeFusedQKNormRoPE(static_cast<char*>(d_input) + offset_bytes,
                         d_out_ref, d_weight,
                         static_cast<int64_t*>(d_pos) + (total_tokens - num_active),
                         d_cache, num_active, num_heads, head_dim, eps, stream_, stride_size,
                         nullptr);
  cudaStreamSynchronize(stream_);

  std::vector<__nv_bfloat16> h_masked(total_size), h_ref(num_active * stride_size);
  cudaMemcpy(h_masked.data(), d_out_masked, total_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ref.data(), d_out_ref, num_active * stride_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

  float max_err = 0;
  int start = (total_tokens - num_active) * stride_size;
  for (int i = 0; i < num_active * stride_size; i++) {
    float a = __bfloat162float(h_masked[start + i]);
    float b = __bfloat162float(h_ref[i]);
    max_err = std::max(max_err, std::abs(a - b));
  }
  EXPECT_LT(max_err, 1e-4f);

  // Verify masked region untouched
  __nv_bfloat16 sentinel;
  memset(&sentinel, 0xAB, sizeof(sentinel));
  int untouched = 0;
  for (int i = 0; i < start; i++) {
    if (memcmp(&h_masked[i], &sentinel, sizeof(__nv_bfloat16)) != 0) untouched++;
  }
  EXPECT_EQ(untouched, 0) << "Masked tokens should not be modified";

  cudaFree(d_input); cudaFree(d_out_masked); cudaFree(d_out_ref);
  cudaFree(d_weight); cudaFree(d_cache); cudaFree(d_pos); cudaFree(d_mask);
}

TEST_F(FusedQKNormRoPETest, Masked_AllActive) {
  const int num_tokens = 512, num_heads = 24, head_dim = 128;
  const float eps = 1e-6f;
  const int stride_size = num_heads * head_dim;
  const int total_size = num_tokens * stride_size;
  const int max_pos = 4096;

  std::mt19937 gen(77);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<__nv_bfloat16> h_input(total_size);
  for (auto& v : h_input) v = __float2bfloat16(dist(gen));
  std::vector<__nv_bfloat16> h_weight(head_dim);
  for (auto& v : h_weight) v = __float2bfloat16(dist(gen));
  std::vector<int64_t> h_pos(num_tokens);
  for (int i = 0; i < num_tokens; i++) h_pos[i] = i;
  std::vector<int64_t> h_mask(num_tokens, 1);
  std::vector<__nv_bfloat16> h_cache(max_pos * head_dim);
  for (auto& v : h_cache) v = __float2bfloat16(dist(gen));

  void *d_input, *d_out_masked, *d_out_nomask, *d_weight, *d_cache, *d_pos, *d_mask;
  cudaMalloc(&d_input, total_size * sizeof(__nv_bfloat16));
  cudaMalloc(&d_out_masked, total_size * sizeof(__nv_bfloat16));
  cudaMalloc(&d_out_nomask, total_size * sizeof(__nv_bfloat16));
  cudaMalloc(&d_weight, head_dim * sizeof(__nv_bfloat16));
  cudaMalloc(&d_cache, max_pos * head_dim * sizeof(__nv_bfloat16));
  cudaMalloc(&d_pos, num_tokens * sizeof(int64_t));
  cudaMalloc(&d_mask, num_tokens * sizeof(int64_t));
  cudaMemcpy(d_input, h_input.data(), total_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight.data(), head_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cache, h_cache.data(), max_pos * head_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos, h_pos.data(), num_tokens * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask.data(), num_tokens * sizeof(int64_t), cudaMemcpyHostToDevice);

  InvokeFusedQKNormRoPE(d_input, d_out_masked, d_weight, d_pos, d_cache,
                         num_tokens, num_heads, head_dim, eps, stream_, stride_size,
                         static_cast<int64_t*>(d_mask));
  InvokeFusedQKNormRoPE(d_input, d_out_nomask, d_weight, d_pos, d_cache,
                         num_tokens, num_heads, head_dim, eps, stream_, stride_size, nullptr);
  cudaStreamSynchronize(stream_);

  std::vector<__nv_bfloat16> h_m(total_size), h_n(total_size);
  cudaMemcpy(h_m.data(), d_out_masked, total_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_n.data(), d_out_nomask, total_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

  float max_err = 0;
  for (int i = 0; i < total_size; i++) {
    max_err = std::max(max_err, std::abs(__bfloat162float(h_m[i]) - __bfloat162float(h_n[i])));
  }
  EXPECT_LT(max_err, 1e-4f);

  cudaFree(d_input); cudaFree(d_out_masked); cudaFree(d_out_nomask);
  cudaFree(d_weight); cudaFree(d_cache); cudaFree(d_pos); cudaFree(d_mask);
}

// Performance
TEST_F(FusedQKNormRoPETest, Perf_Decode_Q) { BenchmarkBoth(1, 24, 128); }
TEST_F(FusedQKNormRoPETest, Perf_Prefill_Q) { BenchmarkBoth(2048, 24, 128); }
TEST_F(FusedQKNormRoPETest, Perf_Prefill_QK) { BenchmarkBoth(2048, 32, 128); }
