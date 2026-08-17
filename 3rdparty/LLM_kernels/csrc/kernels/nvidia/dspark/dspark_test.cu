/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#include "csrc/kernels/nvidia/dspark/dspark.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

namespace {

template <typename T>
T FromFloat(float f);
template <>
__half FromFloat<__half>(float f) {
  return __float2half(f);
}
template <>
__nv_bfloat16 FromFloat<__nv_bfloat16>(float f) {
  return __float2bfloat16(f);
}

template <typename T>
float ToFloat(T v);
template <>
float ToFloat<__half>(__half v) {
  return __half2float(v);
}
template <>
float ToFloat<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

}  // namespace

class DSparkKernelTest : public testing::Test {
 protected:
  void SetUp() override { ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess); }
  void TearDown() override { ASSERT_EQ(cudaStreamDestroy(stream_), cudaSuccess); }

  cudaStream_t stream_;
};

// ---- InvokeDSparkCaptureMeanReduce ----------------------------------------

TEST_F(DSparkKernelTest, CaptureMeanReduceBf16) {
  constexpr int kNumTokens = 4;
  constexpr int kHcMult = 3;
  constexpr int kHiddenDim = 128;
  constexpr int kDstStrideElems = 256;  // dst pitch bigger than hidden_dim to test scatter into wider buffer.
  const int64_t dst_pitch_bytes = kDstStrideElems * sizeof(__nv_bfloat16);

  std::vector<__nv_bfloat16> host_src(kNumTokens * kHcMult * kHiddenDim);
  for (int t = 0; t < kNumTokens; ++t) {
    for (int m = 0; m < kHcMult; ++m) {
      for (int h = 0; h < kHiddenDim; ++h) {
        host_src[(t * kHcMult + m) * kHiddenDim + h] = FromFloat<__nv_bfloat16>(0.01f * (t + 1) * (m + 1) + 0.001f * h);
      }
    }
  }

  void *dev_src = nullptr, *dev_dst = nullptr;
  const size_t dst_bytes = kNumTokens * kDstStrideElems * sizeof(__nv_bfloat16);
  ASSERT_EQ(cudaMalloc(&dev_src, host_src.size() * sizeof(__nv_bfloat16)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_dst, dst_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_src, host_src.data(), host_src.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_dst, 0, dst_bytes), cudaSuccess);

  InvokeDSparkCaptureMeanReduce<__nv_bfloat16>(dev_dst, dev_src, kNumTokens, kHcMult, kHiddenDim, dst_pitch_bytes,
                                               stream_);

  std::vector<__nv_bfloat16> host_dst(kNumTokens * kDstStrideElems);
  ASSERT_EQ(cudaMemcpyAsync(host_dst.data(), dev_dst, dst_bytes, cudaMemcpyDeviceToHost, stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  const float inv_hc = 1.0f / static_cast<float>(kHcMult);
  for (int t = 0; t < kNumTokens; ++t) {
    for (int h = 0; h < kHiddenDim; ++h) {
      float acc = 0.0f;
      for (int m = 0; m < kHcMult; ++m) {
        acc += ToFloat(host_src[(t * kHcMult + m) * kHiddenDim + h]);
      }
      float expected = acc * inv_hc;
      float got = ToFloat(host_dst[t * kDstStrideElems + h]);
      EXPECT_NEAR(expected, got, 5e-3f) << "t=" << t << " h=" << h;
    }
    // Tail region beyond hidden_dim inside pitch must remain zero (not written).
    for (int h = kHiddenDim; h < kDstStrideElems; ++h) {
      EXPECT_EQ(ToFloat(host_dst[t * kDstStrideElems + h]), 0.0f) << "t=" << t << " tail h=" << h;
    }
  }

  cudaFree(dev_src);
  cudaFree(dev_dst);
}

// ---- InvokeDSparkMarkovEmbedLookup ----------------------------------------

TEST_F(DSparkKernelTest, MarkovEmbedLookupBf16Vec8) {
  constexpr int kVocab = 32;
  constexpr int kHidden = 64;
  constexpr int kBatch = 5;

  std::vector<__nv_bfloat16> host_table(kVocab * kHidden);
  for (int v = 0; v < kVocab; ++v) {
    for (int h = 0; h < kHidden; ++h) {
      host_table[v * kHidden + h] = FromFloat<__nv_bfloat16>(0.01f * v + 0.0001f * h);
    }
  }
  std::vector<uint32_t> host_tokens = {0, 3, 7, kVocab - 1, 5};

  void *dev_table = nullptr, *dev_out = nullptr;
  uint32_t *dev_tokens = nullptr;
  const size_t table_bytes = host_table.size() * sizeof(__nv_bfloat16);
  const size_t out_bytes = kBatch * kHidden * sizeof(__nv_bfloat16);
  ASSERT_EQ(cudaMalloc(&dev_table, table_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_out, out_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_tokens, kBatch * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_table, host_table.data(), table_bytes, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_tokens, host_tokens.data(), kBatch * sizeof(uint32_t), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_out, 0xAB, out_bytes), cudaSuccess);

  InvokeDSparkMarkovEmbedLookup<__nv_bfloat16>(dev_out, dev_tokens, dev_table, kBatch, kHidden, stream_);

  std::vector<__nv_bfloat16> host_out(kBatch * kHidden);
  ASSERT_EQ(cudaMemcpyAsync(host_out.data(), dev_out, out_bytes, cudaMemcpyDeviceToHost, stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (int b = 0; b < kBatch; ++b) {
    const uint32_t tok = host_tokens[b];
    for (int h = 0; h < kHidden; ++h) {
      EXPECT_FLOAT_EQ(ToFloat(host_out[b * kHidden + h]), ToFloat(host_table[tok * kHidden + h]))
          << "b=" << b << " h=" << h;
    }
  }

  cudaFree(dev_table);
  cudaFree(dev_out);
  cudaFree(dev_tokens);
}

TEST_F(DSparkKernelTest, MarkovEmbedLookupBf16ScalarFallback) {
  constexpr int kVocab = 8;
  constexpr int kHidden = 17;  // Not vec-aligned -> scalar fallback path.
  constexpr int kBatch = 3;

  std::vector<__nv_bfloat16> host_table(kVocab * kHidden);
  for (int v = 0; v < kVocab; ++v) {
    for (int h = 0; h < kHidden; ++h) {
      host_table[v * kHidden + h] = FromFloat<__nv_bfloat16>(v * 100.0f + h);
    }
  }
  std::vector<uint32_t> host_tokens = {2, 5, 0};

  void *dev_table = nullptr, *dev_out = nullptr;
  uint32_t *dev_tokens = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_table, host_table.size() * sizeof(__nv_bfloat16)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_out, kBatch * kHidden * sizeof(__nv_bfloat16)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_tokens, kBatch * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_table, host_table.data(), host_table.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_tokens, host_tokens.data(), kBatch * sizeof(uint32_t), cudaMemcpyHostToDevice), cudaSuccess);

  InvokeDSparkMarkovEmbedLookup<__nv_bfloat16>(dev_out, dev_tokens, dev_table, kBatch, kHidden, stream_);

  std::vector<__nv_bfloat16> host_out(kBatch * kHidden);
  ASSERT_EQ(cudaMemcpyAsync(host_out.data(), dev_out, host_out.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost,
                            stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (int b = 0; b < kBatch; ++b) {
    for (int h = 0; h < kHidden; ++h) {
      EXPECT_FLOAT_EQ(ToFloat(host_out[b * kHidden + h]), ToFloat(host_table[host_tokens[b] * kHidden + h]));
    }
  }

  cudaFree(dev_table);
  cudaFree(dev_out);
  cudaFree(dev_tokens);
}

// ---- InvokeDSparkLogitsAdd ------------------------------------------------

TEST_F(DSparkKernelTest, LogitsAddBf16Vec4) {
  constexpr int kBatch = 3;
  constexpr int kVocab = 128;

  std::vector<float> host_logits(kBatch * kVocab);
  std::vector<__nv_bfloat16> host_bias(kBatch * kVocab);
  for (int b = 0; b < kBatch; ++b) {
    for (int v = 0; v < kVocab; ++v) {
      host_logits[b * kVocab + v] = 0.1f * (b + 1) * v;
      host_bias[b * kVocab + v] = FromFloat<__nv_bfloat16>(-0.05f * v + 0.01f * b);
    }
  }
  std::vector<float> host_expected(kBatch * kVocab);
  for (size_t i = 0; i < host_expected.size(); ++i) {
    host_expected[i] = host_logits[i] + ToFloat(host_bias[i]);
  }

  void *dev_logits = nullptr, *dev_bias = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_logits, host_logits.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_bias, host_bias.size() * sizeof(__nv_bfloat16)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_logits, host_logits.data(), host_logits.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_bias, host_bias.data(), host_bias.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
            cudaSuccess);

  InvokeDSparkLogitsAdd<__nv_bfloat16>(reinterpret_cast<float *>(dev_logits), dev_bias, kBatch, kVocab, stream_);

  std::vector<float> host_out(kBatch * kVocab);
  ASSERT_EQ(
      cudaMemcpyAsync(host_out.data(), dev_logits, host_out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_),
      cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (size_t i = 0; i < host_out.size(); ++i) {
    EXPECT_NEAR(host_expected[i], host_out[i], 1e-3f) << "i=" << i;
  }

  cudaFree(dev_logits);
  cudaFree(dev_bias);
}

TEST_F(DSparkKernelTest, LogitsAddFp16WithTail) {
  constexpr int kBatch = 2;
  constexpr int kVocab = 127;  // Not vec-aligned -> exercises tail loop.

  std::vector<float> host_logits(kBatch * kVocab);
  std::vector<__half> host_bias(kBatch * kVocab);
  for (int b = 0; b < kBatch; ++b) {
    for (int v = 0; v < kVocab; ++v) {
      host_logits[b * kVocab + v] = 0.001f * v - 0.5f;
      host_bias[b * kVocab + v] = FromFloat<__half>(0.02f * (v - kVocab / 2));
    }
  }
  std::vector<float> host_expected(kBatch * kVocab);
  for (size_t i = 0; i < host_expected.size(); ++i) {
    host_expected[i] = host_logits[i] + ToFloat(host_bias[i]);
  }

  void *dev_logits = nullptr, *dev_bias = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_logits, host_logits.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dev_bias, host_bias.size() * sizeof(__half)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_logits, host_logits.data(), host_logits.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev_bias, host_bias.data(), host_bias.size() * sizeof(__half), cudaMemcpyHostToDevice),
            cudaSuccess);

  InvokeDSparkLogitsAdd<__half>(reinterpret_cast<float *>(dev_logits), dev_bias, kBatch, kVocab, stream_);

  std::vector<float> host_out(kBatch * kVocab);
  ASSERT_EQ(
      cudaMemcpyAsync(host_out.data(), dev_logits, host_out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_),
      cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (size_t i = 0; i < host_out.size(); ++i) {
    EXPECT_NEAR(host_expected[i], host_out[i], 1e-3f) << "i=" << i;
  }

  cudaFree(dev_logits);
  cudaFree(dev_bias);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
