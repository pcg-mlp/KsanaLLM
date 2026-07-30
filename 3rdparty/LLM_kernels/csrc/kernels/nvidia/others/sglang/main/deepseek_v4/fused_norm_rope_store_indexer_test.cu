/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/fused_norm_rope_store_indexer.cuh"

#include <cuda_fp8.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/plan.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

// ─────────────────────────────────────────────────────────────────────────────
// Constants matching the indexer kernel
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int64_t kHeadDim = 128;
static constexpr int64_t kRopeDim = 64;
static constexpr int64_t kNopeDim = 64;             // kHeadDim - kRopeDim
static constexpr int kIndexerTokenDataBytes = 128;  // 128 FP8 values
static constexpr int kIndexerScaleBytes = 4;        // 1 float32 scale
static constexpr float FP8_E4M3_MAX = 448.0f;

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference helpers
// ─────────────────────────────────────────────────────────────────────────────
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

// CPU reference for the indexer fused_norm_rope_store kernel (DefaultForward mode).
// All 128 dims are FP8 quantized with a single float32 per-token scale.
void cpu_indexer_fused_norm_rope_store(const float* input,      // [kHeadDim=128]
                                       const float* weight,     // [kHeadDim=128]
                                       const float* freqs_cis,  // [kRopeDim=64] = [cos(32), sin(32)]
                                       float eps,
                                       uint8_t* fp8_bytes,  // [128]
                                       float* out_scale) {  // [1]
  float normed[kHeadDim];

  // Step 1: RMSNorm with weight
  float sum_sq = 0.0f;
  for (int i = 0; i < kHeadDim; ++i) {
    sum_sq += input[i] * input[i];
  }
  float rms_rcp = 1.0f / std::sqrt(sum_sq / static_cast<float>(kHeadDim) + eps);
  for (int i = 0; i < kHeadDim; ++i) {
    // bf16 roundtrip to match GPU precision
    normed[i] = __bfloat162float(__float2bfloat16(input[i] * rms_rcp * weight[i]));
  }

  // Step 2: RoPE on last 64 dims (split cos/sin layout)
  // freqs_cis layout: [cos_0..cos_31, sin_0..sin_31]
  const float* freq_cos = freqs_cis;
  const float* freq_sin = freqs_cis + kRopeDim / 2;
  for (int p = 0; p < kRopeDim / 2; ++p) {
    float x_re = normed[kNopeDim + 2 * p];
    float x_im = normed[kNopeDim + 2 * p + 1];
    float cos_val = freq_cos[p];
    float sin_val = freq_sin[p];
    normed[kNopeDim + 2 * p] = __bfloat162float(__float2bfloat16(x_re * cos_val - x_im * sin_val));
    normed[kNopeDim + 2 * p + 1] = __bfloat162float(__float2bfloat16(x_re * sin_val + x_im * cos_val));
  }

  // Step 3: FP8 quantize all 128 dims with single float32 scale
  float abs_max = 0.0f;
  for (int i = 0; i < kHeadDim; ++i) {
    abs_max = std::fmax(abs_max, std::fabs(normed[i]));
  }

  float scale = std::fmax(1e-4f, abs_max) / FP8_E4M3_MAX;
  float inv_scale = 1.0f / scale;
  *out_scale = scale;

  for (int i = 0; i < kHeadDim; ++i) {
    float scaled = normed[i] * inv_scale;
    scaled = std::fmin(std::fmax(scaled, -FP8_E4M3_MAX), FP8_E4M3_MAX);
    fp8_bytes[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
  }
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Test fixture
// ─────────────────────────────────────────────────────────────────────────────
class FusedNormRopeStoreIndexerTest : public NvidiaTestSuitBase {
 protected:
  using NvidiaTestSuitBase::stream;

  static constexpr int64_t PageStride(uint32_t page_bits) {
    int64_t page_size = static_cast<int64_t>(1u << page_bits);
    return kIndexerTokenDataBytes * page_size + kIndexerScaleBytes * page_size;
  }

  // Verify DefaultForward correctness for given num_works.
  void TestDefaultForwardCorrectness(uint32_t num_works, uint32_t page_bits) {
    const float eps = 1e-5f;
    const int64_t max_pos = 128;
    const uint32_t page_size = 1u << page_bits;
    const int64_t page_stride = PageStride(page_bits);
    constexpr uint8_t kFill = 0xAA;

    std::mt19937 rng(42u);

    // ── Prepare host data ──
    const size_t in_elem = static_cast<size_t>(num_works) * kHeadDim;
    const size_t w_elem = static_cast<size_t>(kHeadDim);
    const size_t f_elem = static_cast<size_t>(max_pos) * kRopeDim;

    std::vector<float> in_f(in_elem), w_f(w_elem), freq_f(f_elem);
    FillRandomBF16(&in_f, &rng);
    FillRandomBF16(&w_f, &rng, 0.5f, 1.5f);
    FillRandomBF16(&freq_f, &rng);

    std::vector<__nv_bfloat16> in_h, w_h, freq_h;
    HostFloatToBF16(in_f, &in_h);
    HostFloatToBF16(w_f, &w_h);
    HostFloatToBF16(freq_f, &freq_h);

    // Re-derive floats from bf16 to ensure CPU ref uses same precision
    for (size_t i = 0; i < in_f.size(); ++i) in_f[i] = __bfloat162float(in_h[i]);
    for (size_t i = 0; i < w_f.size(); ++i) w_f[i] = __bfloat162float(w_h[i]);
    for (size_t i = 0; i < freq_f.size(); ++i) freq_f[i] = __bfloat162float(freq_h[i]);

    // Positions and indices -- randomly set ~25% to index=-1 (skip store)
    std::vector<int64_t> h_positions(num_works);
    std::vector<int32_t> h_indices(num_works);
    std::bernoulli_distribution skip_dist(0.25);
    int32_t next_slot = 0;
    for (uint32_t i = 0; i < num_works; ++i) {
      h_positions[i] = static_cast<int64_t>(i % max_pos);
      if (num_works > 1 && skip_dist(rng)) {
        h_indices[i] = -1;  // skip
      } else {
        h_indices[i] = next_slot++;
      }
    }
    // Ensure at least one valid index exists
    if (next_slot == 0) {
      h_indices[0] = next_slot++;
    }
    const uint32_t num_valid_slots = static_cast<uint32_t>(next_slot);

    // ── Allocate device memory ──
    uint32_t num_pages = (num_valid_slots + page_size - 1) / page_size;
    size_t cache_bytes = static_cast<size_t>(num_pages) * page_stride;

    BufferMeta d_input = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {in_elem});
    BufferMeta d_weight = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {w_elem});
    BufferMeta d_freqs = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {f_elem});
    BufferMeta d_positions = CreateBuffer<int64_t>(MEMORY_GPU, {num_works});
    BufferMeta d_indices = CreateBuffer<int32_t>(MEMORY_GPU, {num_works});
    BufferMeta d_cache = CreateBuffer<uint8_t>(MEMORY_GPU, {cache_bytes});

    // Fill cache with sentinel so we can verify skipped slots remain untouched
    CHECK_NVIDIA_CUDA_ERROR(cudaMemset(d_cache.data_ptr, kFill, cache_bytes));

    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_input.data_ptr, in_h.data(), in_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_weight.data_ptr, w_h.data(), w_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_freqs.data_ptr, freq_h.data(), f_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_positions.data_ptr, h_positions.data(), num_works * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_indices.data_ptr, h_indices.data(), num_works * sizeof(int32_t), cudaMemcpyHostToDevice));

    // ── Launch kernel ──
    indexer_fused_norm_rope_store<__nv_bfloat16, kHeadDim, kRopeDim>(
        d_input.data_ptr, d_weight.data_ptr, d_positions.data_ptr, d_freqs.data_ptr, num_works,
        ForwardMode::DefaultForward, eps, /*compress_ratio=*/4u, d_cache.data_ptr,
        reinterpret_cast<int32_t*>(d_indices.data_ptr), page_stride, page_bits, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // ── Copy results back ──
    std::vector<uint8_t> h_cache(cache_bytes);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_cache.data(), d_cache.data_ptr, cache_bytes, cudaMemcpyDeviceToHost));

    // ── Verify each work item ──
    for (uint32_t w = 0; w < num_works; ++w) {
      int32_t index = h_indices[w];
      if (index == -1) continue;  // skipped items don't write

      const float* input_head = &in_f[w * kHeadDim];
      const float* freq_ptr = &freq_f[h_positions[w] * kRopeDim];

      uint8_t expected_fp8[kHeadDim];
      float expected_scale;
      cpu_indexer_fused_norm_rope_store(input_head, w_f.data(), freq_ptr, eps, expected_fp8, &expected_scale);

      // Locate in cache
      int32_t page = index >> page_bits;
      int32_t offset = index & ((1 << page_bits) - 1);
      const uint8_t* page_ptr = &h_cache[page * page_stride];
      const uint8_t* slot_ptr = page_ptr + offset * kIndexerTokenDataBytes;

      // Verify FP8 bytes [0, 128)
      for (int i = 0; i < kHeadDim; ++i) {
        EXPECT_EQ(slot_ptr[i], expected_fp8[i]) << "FP8 mismatch at work=" << w << " dim=" << i;
      }

      // Verify float32 scale
      const float* scale_ptr = reinterpret_cast<const float*>(
          page_ptr + static_cast<int64_t>(kIndexerTokenDataBytes) * page_size + offset * kIndexerScaleBytes);
      EXPECT_NEAR(*scale_ptr, expected_scale, expected_scale * 1e-5f) << "Scale mismatch at work=" << w;
    }

    // ── Cleanup ──
    DeleteBuffer(d_input);
    DeleteBuffer(d_weight);
    DeleteBuffer(d_freqs);
    DeleteBuffer(d_positions);
    DeleteBuffer(d_indices);
    DeleteBuffer(d_cache);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Correctness tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(FusedNormRopeStoreIndexerTest, DefaultForwardSingleToken) {
  TestDefaultForwardCorrectness(/*num_works=*/1, /*page_bits=*/1);
}

TEST_F(FusedNormRopeStoreIndexerTest, DefaultForwardMultiToken) {
  TestDefaultForwardCorrectness(/*num_works=*/8, /*page_bits=*/8);
}

TEST_F(FusedNormRopeStoreIndexerTest, DefaultForwardPageBits6) {
  TestDefaultForwardCorrectness(/*num_works=*/16, /*page_bits=*/6);
}

// ─────────────────────────────────────────────────────────────────────────────
// Performance test (disabled by default)
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(FusedNormRopeStoreIndexerTest, DISABLED_PerfTest) {
  const std::vector<uint32_t> token_nums = {1, 8, 32, 128, 512, 2048};
  constexpr float eps = 1e-5f;
  constexpr int64_t max_pos = 8192;
  constexpr uint32_t page_bits = 6;
  constexpr uint32_t page_size = 1u << page_bits;
  const int64_t page_stride = PageStride(page_bits);

  std::cout << "\n=== FusedNormRopeStoreIndexer Performance ===\n";
  std::cout << "HeadDim=" << kHeadDim << ", RopeDim=" << kRopeDim << ", PageBits=" << page_bits << "\n\n";

  // Allocate invariant buffers once at max size
  constexpr uint32_t max_works = 2048;
  constexpr size_t max_in_elem = static_cast<size_t>(max_works) * kHeadDim;
  uint32_t max_pages = (max_works + page_size - 1) / page_size;
  size_t max_cache_bytes = static_cast<size_t>(max_pages) * page_stride;

  BufferMeta d_input = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {max_in_elem}, /*random=*/true);
  BufferMeta d_weight = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {static_cast<size_t>(kHeadDim)}, /*random=*/true);
  BufferMeta d_freqs =
      CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {static_cast<size_t>(max_pos * kRopeDim)}, /*random=*/true);
  BufferMeta d_positions = CreateBuffer<int64_t>(MEMORY_GPU, {max_works});
  BufferMeta d_indices = CreateBuffer<int32_t>(MEMORY_GPU, {max_works});
  BufferMeta d_cache = CreateBuffer<uint8_t>(MEMORY_GPU, {max_cache_bytes});

  for (uint32_t num_works : token_nums) {
    // Fill positions and indices
    std::vector<int64_t> h_pos(num_works);
    std::vector<int32_t> h_idx(num_works);
    for (uint32_t i = 0; i < num_works; ++i) {
      h_pos[i] = static_cast<int64_t>(i % max_pos);
      h_idx[i] = static_cast<int32_t>(i);
    }
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_positions.data_ptr, h_pos.data(), num_works * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_indices.data_ptr, h_idx.data(), num_works * sizeof(int32_t), cudaMemcpyHostToDevice));

    const int warmups = 5;
    const int iterations = 20;
    auto cuda_run = [&]() {
      indexer_fused_norm_rope_store<__nv_bfloat16, kHeadDim, kRopeDim>(
          d_input.data_ptr, d_weight.data_ptr, d_positions.data_ptr, d_freqs.data_ptr, num_works,
          ForwardMode::DefaultForward, eps, /*compress_ratio=*/4u, d_cache.data_ptr,
          reinterpret_cast<int32_t*>(d_indices.data_ptr), page_stride, page_bits, stream);
    };
    float elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, warmups, iterations);

    double input_bytes = static_cast<double>(num_works) * kHeadDim * sizeof(__nv_bfloat16);
    double weight_bytes = static_cast<double>(kHeadDim) * sizeof(__nv_bfloat16);
    double freqs_bytes = static_cast<double>(num_works) * kRopeDim * sizeof(__nv_bfloat16);
    double cache_write_bytes = static_cast<double>(num_works) * (kIndexerTokenDataBytes + kIndexerScaleBytes);
    double total_bytes = input_bytes + weight_bytes + freqs_bytes + cache_write_bytes;
    double bandwidth_gbps = total_bytes / (elapsed_ms * 1e-3) / 1e9;

    std::cout << "  Tokens: " << std::setw(5) << num_works << ", Time: " << std::fixed << std::setprecision(4)
              << elapsed_ms << " ms"
              << ", BW: " << std::setprecision(1) << bandwidth_gbps << " GB/s" << std::endl;
  }

  DeleteBuffer(d_input);
  DeleteBuffer(d_weight);
  DeleteBuffer(d_freqs);
  DeleteBuffer(d_positions);
  DeleteBuffer(d_indices);
  DeleteBuffer(d_cache);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
