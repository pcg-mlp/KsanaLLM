/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Unit tests for the fused K indexer LayerNorm + RoPE + FP8 quant + paged store kernel.
 */

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v32/indexer_k.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

static constexpr int64_t kHeadDim = 128;
static constexpr int64_t kRopeDim = 64;
static constexpr int kIndexerTokenDataBytes = 128;
static constexpr int kIndexerScaleBytes = 4;
static constexpr float FP8_E4M3_MAX = 448.0f;

namespace {

void FillRandomBF16(std::vector<float>* v, std::mt19937* rng, float lo = -1.0f, float hi = 1.0f) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& x : *v) {
    x = __bfloat162float(__float2bfloat16(dist(*rng)));
  }
}

void HostFloatToBF16(const std::vector<float>& src, std::vector<__nv_bfloat16>* dst) {
  dst->resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) (*dst)[i] = __float2bfloat16(src[i]);
}

// CPU reference matching the kernel's lane-major layout.
// LayerNorm (mean-centered) + RoPE on rope lanes [0..15] + FP8 quant.
void cpu_fused_k_indexer_norm_rope(const float* input, const float* gamma, const float* beta, const float* freqs_cis,
                                   float eps, bool is_neox, uint8_t* out_fp8, float* out_scale) {
  float head[kHeadDim];

  // Part 1: LayerNorm
  float sum = 0.0f;
  for (int i = 0; i < kHeadDim; ++i) sum += input[i];
  float mean = sum / static_cast<float>(kHeadDim);

  float var = 0.0f;
  for (int i = 0; i < kHeadDim; ++i) {
    float centered = input[i] - mean;
    var += centered * centered;
  }
  float inv_std = 1.0f / std::sqrt(var / static_cast<float>(kHeadDim) + eps);
  for (int i = 0; i < kHeadDim; ++i) {
    head[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
  }

  // Part 2: RoPE on rope lanes 0..15 (first 64 dims). Lane l owns head[l*4..l*4+3].
  const float* freq_cos = freqs_cis;
  const float* freq_sin = freqs_cis + kRopeDim / 2;
  constexpr uint32_t kRopeLanes = kRopeDim / 4;  // = 16
  if (is_neox) {
    // Lanes [0..7] = real, [8..15] = imag. freq_idx = lane * 4.
    constexpr uint32_t half = kRopeLanes / 2;  // = 8
    for (uint32_t rl = 0; rl < half; ++rl) {
      uint32_t real_lane = rl;
      uint32_t imag_lane = rl + half;
      uint32_t freq_idx = rl * 4;
      for (int i = 0; i < 4; ++i) {
        float x_real = head[real_lane * 4 + i];
        float x_imag = head[imag_lane * 4 + i];
        float f_cos = freq_cos[freq_idx + i];
        float f_sin = freq_sin[freq_idx + i];
        head[real_lane * 4 + i] = x_real * f_cos - x_imag * f_sin;
        head[imag_lane * 4 + i] = x_real * f_sin + x_imag * f_cos;
      }
    }
  } else {
    // GPT-J interleaved: 4 elems per lane = 2 (real, imag) pairs.
    for (uint32_t rl = 0; rl < kRopeLanes; ++rl) {
      uint32_t freq_idx = rl * 2;
      float x_real = head[rl * 4 + 0];
      float x_imag = head[rl * 4 + 1];
      float y_real = head[rl * 4 + 2];
      float y_imag = head[rl * 4 + 3];
      float f_cos_0 = freq_cos[freq_idx];
      float f_sin_0 = freq_sin[freq_idx];
      float f_cos_1 = freq_cos[freq_idx + 1];
      float f_sin_1 = freq_sin[freq_idx + 1];
      head[rl * 4 + 0] = x_real * f_cos_0 - x_imag * f_sin_0;
      head[rl * 4 + 1] = x_real * f_sin_0 + x_imag * f_cos_0;
      head[rl * 4 + 2] = y_real * f_cos_1 - y_imag * f_sin_1;
      head[rl * 4 + 3] = y_real * f_sin_1 + y_imag * f_cos_1;
    }
  }

  // Part 3: FP8 quant with per-token scale.
  float abs_max = 0.0f;
  for (int i = 0; i < kHeadDim; ++i) abs_max = std::fmax(abs_max, std::fabs(head[i]));
  float scale = std::fmax(1e-4f, abs_max) / FP8_E4M3_MAX;
  float inv_scale = 1.0f / scale;
  *out_scale = scale;

  for (int i = 0; i < kHeadDim; ++i) {
    float scaled = head[i] * inv_scale;
    scaled = std::fmin(std::fmax(scaled, -FP8_E4M3_MAX), FP8_E4M3_MAX);
    out_fp8[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
  }
}

}  // namespace

class FusedKIndexerNormRopeStoreTest : public NvidiaTestSuitBase {
 protected:
  using NvidiaTestSuitBase::stream;

  static int64_t PageStride(uint32_t page_bits) {
    int64_t page_size = static_cast<int64_t>(1u << page_bits);
    return kIndexerTokenDataBytes * page_size + kIndexerScaleBytes * page_size;
  }

  void TestCorrectness(uint32_t batch_size, uint32_t page_bits, bool is_neox) {
    const float eps = 1e-5f;
    const int64_t max_pos = 128;
    const uint32_t page_size = 1u << page_bits;
    const int64_t page_stride = PageStride(page_bits);
    constexpr uint8_t kFill = 0xAA;

    std::mt19937 rng(42u);

    const size_t in_elem = static_cast<size_t>(batch_size) * kHeadDim;
    const size_t w_elem = static_cast<size_t>(kHeadDim);
    const size_t f_elem = static_cast<size_t>(max_pos) * kRopeDim;

    std::vector<float> in_f(in_elem), gamma_f(w_elem), beta_f(w_elem), freq_f(f_elem);
    FillRandomBF16(&in_f, &rng);
    FillRandomBF16(&gamma_f, &rng, 0.5f, 1.5f);
    FillRandomBF16(&beta_f, &rng, -0.1f, 0.1f);
    FillRandomBF16(&freq_f, &rng);

    std::vector<__nv_bfloat16> in_h, gamma_h, beta_h, freq_h;
    HostFloatToBF16(in_f, &in_h);
    HostFloatToBF16(gamma_f, &gamma_h);
    HostFloatToBF16(beta_f, &beta_h);
    HostFloatToBF16(freq_f, &freq_h);

    for (size_t i = 0; i < in_f.size(); ++i) in_f[i] = __bfloat162float(in_h[i]);
    for (size_t i = 0; i < gamma_f.size(); ++i) gamma_f[i] = __bfloat162float(gamma_h[i]);
    for (size_t i = 0; i < beta_f.size(); ++i) beta_f[i] = __bfloat162float(beta_h[i]);
    for (size_t i = 0; i < freq_f.size(); ++i) freq_f[i] = __bfloat162float(freq_h[i]);

    // Positions and indices; ~25% marked -1 to exercise the skip path.
    std::vector<int64_t> h_positions(batch_size);
    std::vector<int32_t> h_indices(batch_size);
    std::bernoulli_distribution skip_dist(0.25);
    int32_t next_slot = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      h_positions[i] = static_cast<int64_t>(i % max_pos);
      if (batch_size > 1 && skip_dist(rng)) {
        h_indices[i] = -1;
      } else {
        h_indices[i] = next_slot++;
      }
    }
    if (next_slot == 0) h_indices[0] = next_slot++;
    const uint32_t num_valid = static_cast<uint32_t>(next_slot);

    uint32_t num_pages = (num_valid + page_size - 1) / page_size;
    size_t cache_bytes = static_cast<size_t>(num_pages) * page_stride;

    BufferMeta d_input = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {in_elem});
    BufferMeta d_gamma = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {w_elem});
    BufferMeta d_beta = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {w_elem});
    BufferMeta d_freqs = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {f_elem});
    BufferMeta d_positions = CreateBuffer<int64_t>(MEMORY_GPU, {batch_size});
    BufferMeta d_indices = CreateBuffer<int32_t>(MEMORY_GPU, {batch_size});
    BufferMeta d_cache = CreateBuffer<uint8_t>(MEMORY_GPU, {cache_bytes});

    CHECK_NVIDIA_CUDA_ERROR(cudaMemset(d_cache.data_ptr, kFill, cache_bytes));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_input.data_ptr, in_h.data(), in_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_gamma.data_ptr, gamma_h.data(), w_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_beta.data_ptr, beta_h.data(), w_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_freqs.data_ptr, freq_h.data(), f_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_positions.data_ptr, h_positions.data(), batch_size * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_indices.data_ptr, h_indices.data(), batch_size * sizeof(int32_t), cudaMemcpyHostToDevice));

    fused_k_indexer_norm_rope_store(d_input.data_ptr, d_cache.data_ptr, d_indices.data_ptr, d_gamma.data_ptr,
                                    d_beta.data_ptr, d_freqs.data_ptr, d_positions.data_ptr,
                                    /*k_input_stride_batch=*/kHeadDim, batch_size, page_bits, page_stride, eps, stream,
                                    is_neox);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    std::vector<uint8_t> h_cache(cache_bytes);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_cache.data(), d_cache.data_ptr, cache_bytes, cudaMemcpyDeviceToHost));

    for (uint32_t b = 0; b < batch_size; ++b) {
      int32_t index = h_indices[b];
      if (index == -1) continue;

      const float* input_ptr = &in_f[b * kHeadDim];
      const float* freq_ptr = &freq_f[h_positions[b] * kRopeDim];

      uint8_t expected_fp8[kHeadDim];
      float expected_scale;
      cpu_fused_k_indexer_norm_rope(input_ptr, gamma_f.data(), beta_f.data(), freq_ptr, eps, is_neox, expected_fp8,
                                    &expected_scale);

      int32_t page = index >> page_bits;
      int32_t offset = index & ((1 << page_bits) - 1);
      const uint8_t* page_ptr = &h_cache[page * page_stride];
      const uint8_t* slot_ptr = page_ptr + offset * kIndexerTokenDataBytes;

      for (int i = 0; i < kHeadDim; ++i) {
        int diff = static_cast<int>(slot_ptr[i]) - static_cast<int>(expected_fp8[i]);
        EXPECT_LE(std::abs(diff), 1) << "FP8 mismatch >1 ULP at batch=" << b << " dim=" << i
                                     << " actual=" << static_cast<int>(slot_ptr[i])
                                     << " expected=" << static_cast<int>(expected_fp8[i]) << " is_neox=" << is_neox;
      }

      const float* scale_ptr = reinterpret_cast<const float*>(
          page_ptr + static_cast<int64_t>(kIndexerTokenDataBytes) * page_size + offset * kIndexerScaleBytes);
      EXPECT_NEAR(*scale_ptr, expected_scale, std::fabs(expected_scale) * 1e-3f)
          << "Scale mismatch at batch=" << b << " is_neox=" << is_neox;
    }

    DeleteBuffer(d_input);
    DeleteBuffer(d_gamma);
    DeleteBuffer(d_beta);
    DeleteBuffer(d_freqs);
    DeleteBuffer(d_positions);
    DeleteBuffer(d_indices);
    DeleteBuffer(d_cache);
  }
};

TEST_F(FusedKIndexerNormRopeStoreTest, InterleavedSingleToken) {
  TestCorrectness(/*batch_size=*/1, /*page_bits=*/1, /*is_neox=*/false);
}

TEST_F(FusedKIndexerNormRopeStoreTest, InterleavedMultiToken) {
  TestCorrectness(/*batch_size=*/8, /*page_bits=*/6, /*is_neox=*/false);
}

TEST_F(FusedKIndexerNormRopeStoreTest, InterleavedPageBits8) {
  TestCorrectness(/*batch_size=*/64, /*page_bits=*/8, /*is_neox=*/false);
}

TEST_F(FusedKIndexerNormRopeStoreTest, NeoxSingleToken) {
  TestCorrectness(/*batch_size=*/1, /*page_bits=*/1, /*is_neox=*/true);
}

TEST_F(FusedKIndexerNormRopeStoreTest, NeoxMultiToken) {
  TestCorrectness(/*batch_size=*/16, /*page_bits=*/6, /*is_neox=*/true);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
