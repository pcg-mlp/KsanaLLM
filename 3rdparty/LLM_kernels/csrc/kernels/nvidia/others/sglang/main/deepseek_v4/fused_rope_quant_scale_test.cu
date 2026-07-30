/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Unit tests for the fused RoPE + FP8 quantization + weight scaling kernel.
 */

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/fused_rope_quant_scale.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

static constexpr int64_t kHeadDim = 128;
static constexpr int64_t kRopeDim = 64;
static constexpr int64_t kNopeDim = 64;
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

// CPU reference for one (token, head): RoPE + FP8 quant + weight scale.
// freqs_cis: split [cos|sin] layout, bf16-rounded float.
// rope_offset: element offset within head where the 64 rope dims start.
// is_neox: if true, split-half layout within the rope region;
//          if false, interleaved layout within the rope region.
void cpu_fused_rope_quant_scale(const float* q_head, const float* freqs_cis, float weight_val, float combined_scale,
                                uint32_t rope_offset, bool is_neox, uint8_t* out_fp8, float* out_scale) {
  // Work in lane-major order: [32 lanes][4 elems] to match kernel's data layout.
  // Lane `l` owns elements [l*4, l*4+3].
  float head[kHeadDim];
  for (int i = 0; i < kHeadDim; ++i) head[i] = q_head[i];

  // RoPE (applied in lane-major order, matching kernel behavior)
  const uint32_t rope_lane_start = rope_offset / 4;

  const float* freq_cos = freqs_cis;
  const float* freq_sin = freqs_cis + kRopeDim / 2;

  if (is_neox) {
    // NeoX style: within the rope region, first half lanes = real, second half lanes = imag.
    // Rope region (in lane-major): lanes [rope_lane_start..rope_lane_end-1]
    // Real lanes: [rope_lane_start..rope_lane_start+7], Imag lanes: [rope_lane_start+8..rope_lane_end-1]
    constexpr uint32_t half_rope_lanes = 8;
    for (uint32_t rl = 0; rl < half_rope_lanes; ++rl) {
      uint32_t real_lane = rope_lane_start + rl;
      uint32_t imag_lane = rope_lane_start + half_rope_lanes + rl;
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
    // GPT-J interleaved style: 4 elems per lane = 2 (real, imag) pairs.
    for (uint32_t rl = 0; rl < 16; ++rl) {
      uint32_t lane = rope_lane_start + rl;
      uint32_t freq_idx = rl * 2;
      float x_real = head[lane * 4 + 0];
      float x_imag = head[lane * 4 + 1];
      float y_real = head[lane * 4 + 2];
      float y_imag = head[lane * 4 + 3];
      float f_cos_0 = freq_cos[freq_idx];
      float f_sin_0 = freq_sin[freq_idx];
      float f_cos_1 = freq_cos[freq_idx + 1];
      float f_sin_1 = freq_sin[freq_idx + 1];
      head[lane * 4 + 0] = x_real * f_cos_0 - x_imag * f_sin_0;
      head[lane * 4 + 1] = x_real * f_sin_0 + x_imag * f_cos_0;
      head[lane * 4 + 2] = y_real * f_cos_1 - y_imag * f_sin_1;
      head[lane * 4 + 3] = y_real * f_sin_1 + y_imag * f_cos_1;
    }
  }

  // FP8 quantize
  float abs_max = 0.0f;
  for (int i = 0; i < kHeadDim; ++i) {
    abs_max = std::fmax(abs_max, std::fabs(head[i]));
  }

  float q_scale = std::fmax(1e-4f, abs_max) / FP8_E4M3_MAX;
  float inv_scale = 1.0f / q_scale;

  for (int i = 0; i < kHeadDim; ++i) {
    float scaled = head[i] * inv_scale;
    scaled = std::fmin(std::fmax(scaled, -FP8_E4M3_MAX), FP8_E4M3_MAX);
    out_fp8[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
  }

  // Combined scale
  *out_scale = weight_val * combined_scale * q_scale;
}

}  // namespace

class FusedRopeQuantScaleTest : public NvidiaTestSuitBase {
 protected:
  using NvidiaTestSuitBase::stream;

  void TestCorrectness(uint32_t batch_size, uint32_t num_heads, uint32_t rope_offset, bool is_neox) {
    const int64_t max_pos = 256;
    const float n_heads_inv_sqrt = 1.0f / std::sqrt(static_cast<float>(num_heads));
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    const float combined_scale = n_heads_inv_sqrt * softmax_scale;

    std::mt19937 rng(42u);

    // Prepare host data
    const size_t q_elem = static_cast<size_t>(batch_size) * num_heads * kHeadDim;
    const size_t w_elem = static_cast<size_t>(batch_size) * num_heads;
    const size_t f_elem = static_cast<size_t>(max_pos) * kRopeDim;

    std::vector<float> q_f(q_elem), w_f(w_elem), freq_f(f_elem);
    FillRandomBF16(&q_f, &rng);
    FillRandomBF16(&w_f, &rng, -0.5f, 0.5f);
    FillRandomBF16(&freq_f, &rng);

    std::vector<__nv_bfloat16> q_h, freq_h;
    HostFloatToBF16(q_f, &q_h);
    HostFloatToBF16(freq_f, &freq_h);

    // Re-derive floats from bf16 to match kernel precision
    for (size_t i = 0; i < q_f.size(); ++i) q_f[i] = __bfloat162float(q_h[i]);
    for (size_t i = 0; i < freq_f.size(); ++i) freq_f[i] = __bfloat162float(freq_h[i]);

    std::vector<int64_t> h_positions(batch_size);
    for (uint32_t i = 0; i < batch_size; ++i) {
      h_positions[i] = static_cast<int64_t>(i % max_pos);
    }

    // Allocate device memory
    BufferMeta d_q = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {q_elem});
    BufferMeta d_weights = CreateBuffer<float>(MEMORY_GPU, {w_elem});
    BufferMeta d_freqs = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {f_elem});
    BufferMeta d_positions = CreateBuffer<int64_t>(MEMORY_GPU, {static_cast<size_t>(batch_size)});
    BufferMeta d_output_q_fp8 =
        CreateBuffer<uint8_t>(MEMORY_GPU, {static_cast<size_t>(batch_size * num_heads * kHeadDim)});
    BufferMeta d_output_scale = CreateBuffer<float>(MEMORY_GPU, {w_elem});

    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_q.data_ptr, q_h.data(), q_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_weights.data_ptr, w_f.data(), w_elem * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_freqs.data_ptr, freq_h.data(), f_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_positions.data_ptr, h_positions.data(), batch_size * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Launch kernel
    fused_rope_quant_scale(d_q.data_ptr, d_weights.data_ptr, d_freqs.data_ptr, d_positions.data_ptr,
                           d_output_q_fp8.data_ptr, static_cast<float*>(d_output_scale.data_ptr), n_heads_inv_sqrt,
                           softmax_scale, num_heads, batch_size, rope_offset, stream, is_neox);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy results back
    std::vector<uint8_t> h_output_q_fp8(batch_size * num_heads * kHeadDim);
    std::vector<float> h_output_scale(batch_size * num_heads);
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(h_output_q_fp8.data(), d_output_q_fp8.data_ptr, h_output_q_fp8.size(), cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_output_scale.data(), d_output_scale.data_ptr,
                                       h_output_scale.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify each (token, head)
    for (uint32_t b = 0; b < batch_size; ++b) {
      for (uint32_t h = 0; h < num_heads; ++h) {
        const size_t head_offset = (b * num_heads + h) * kHeadDim;
        const float* q_head = &q_f[head_offset];
        const float* freq_ptr = &freq_f[h_positions[b] * kRopeDim];
        const float weight_val = w_f[b * num_heads + h];

        uint8_t expected_fp8[kHeadDim];
        float expected_scale;
        cpu_fused_rope_quant_scale(q_head, freq_ptr, weight_val, combined_scale, rope_offset, is_neox, expected_fp8,
                                   &expected_scale);

        // The kernel uses fp32 arithmetic throughout (after bf16 load),
        // so we expect very tight FP8 agreement.
        // Allow ±1 ULP for hardware vs software fp8 rounding.
        const uint8_t* actual_fp8 = &h_output_q_fp8[head_offset];
        for (int i = 0; i < kHeadDim; ++i) {
          int diff = static_cast<int>(actual_fp8[i]) - static_cast<int>(expected_fp8[i]);
          EXPECT_LE(std::abs(diff), 1) << "FP8 mismatch >1 ULP at batch=" << b << " head=" << h << " dim=" << i
                                       << " actual=" << static_cast<int>(actual_fp8[i])
                                       << " expected=" << static_cast<int>(expected_fp8[i]) << " is_neox=" << is_neox
                                       << " rope_offset=" << rope_offset;
        }

        // Verify combined scale
        float actual_scale = h_output_scale[b * num_heads + h];
        EXPECT_NEAR(actual_scale, expected_scale, std::fabs(expected_scale) * 1e-3f)
            << "Scale mismatch at batch=" << b << " head=" << h << " is_neox=" << is_neox;
      }
    }

    DeleteBuffer(d_q);
    DeleteBuffer(d_weights);
    DeleteBuffer(d_freqs);
    DeleteBuffer(d_positions);
    DeleteBuffer(d_output_q_fp8);
    DeleteBuffer(d_output_scale);
  }
};

// Interleaved (is_neox=false) tests with rope_offset=64 (C4 indexer: [nope, rope] layout)
TEST_F(FusedRopeQuantScaleTest, SingleTokenSingleHead) {
  TestCorrectness(/*batch_size=*/1, /*num_heads=*/1, /*rope_offset=*/static_cast<uint32_t>(kNopeDim),
                  /*is_neox=*/false);
}

TEST_F(FusedRopeQuantScaleTest, MultiTokenMultiHead) {
  TestCorrectness(/*batch_size=*/8, /*num_heads=*/16, /*rope_offset=*/static_cast<uint32_t>(kNopeDim),
                  /*is_neox=*/false);
}

TEST_F(FusedRopeQuantScaleTest, LargeBatch) {
  TestCorrectness(/*batch_size=*/128, /*num_heads=*/128, /*rope_offset=*/static_cast<uint32_t>(kNopeDim),
                  /*is_neox=*/false);
}

// Interleaved with rope_offset=0 (DSA indexer: [rope, nope] layout)
TEST_F(FusedRopeQuantScaleTest, InterleavedRopeOffset0) {
  TestCorrectness(/*batch_size=*/8, /*num_heads=*/16, /*rope_offset=*/0, /*is_neox=*/false);
}

// NeoX (is_neox=true) tests with rope_offset=0 (DSA indexer: [rope, nope] layout)
TEST_F(FusedRopeQuantScaleTest, NeoxRopeOffset0) {
  TestCorrectness(/*batch_size=*/8, /*num_heads=*/16, /*rope_offset=*/0, /*is_neox=*/true);
}

// NeoX with rope_offset=64 (C4 indexer: [nope, rope] layout)
TEST_F(FusedRopeQuantScaleTest, NeoxRopeOffset64) {
  TestCorrectness(/*batch_size=*/8, /*num_heads=*/16, /*rope_offset=*/static_cast<uint32_t>(kNopeDim),
                  /*is_neox=*/true);
}

// NeoX large batch
TEST_F(FusedRopeQuantScaleTest, NeoxLargeBatch) {
  TestCorrectness(/*batch_size=*/128, /*num_heads=*/128, /*rope_offset=*/0, /*is_neox=*/true);
}

// NeoX with page_bits=8 (256 tokens/page): batch spans full position range.
TEST_F(FusedRopeQuantScaleTest, NeoxPageBits8RopeOffset0) {
  TestCorrectness(/*batch_size=*/256, /*num_heads=*/16, /*rope_offset=*/0, /*is_neox=*/true);
}

TEST_F(FusedRopeQuantScaleTest, NeoxPageBits8RopeOffset64) {
  TestCorrectness(/*batch_size=*/256, /*num_heads=*/16, /*rope_offset=*/static_cast<uint32_t>(kNopeDim),
                  /*is_neox=*/true);
}

TEST_F(FusedRopeQuantScaleTest, DISABLED_PerfTest) {
  const std::vector<uint32_t> batch_sizes = {1, 8, 32, 128, 512, 2048};
  constexpr uint32_t num_heads = 128;
  constexpr int64_t max_pos = 8192;

  const float n_heads_inv_sqrt = 1.0f / std::sqrt(static_cast<float>(num_heads));
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

  std::cout << "\n=== FusedRopeQuantScale Performance ===\n";
  std::cout << "HeadDim=" << kHeadDim << ", RopeDim=" << kRopeDim << ", NumHeads=" << num_heads << "\n\n";

  // Allocate buffers at max size
  constexpr uint32_t max_batch = 2048;
  const size_t max_q_elem = static_cast<size_t>(max_batch) * num_heads * kHeadDim;
  const size_t max_w_elem = static_cast<size_t>(max_batch) * num_heads;

  BufferMeta d_q = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {max_q_elem}, /*random=*/true);
  BufferMeta d_weights = CreateBuffer<float>(MEMORY_GPU, {max_w_elem}, /*random=*/true);
  BufferMeta d_freqs =
      CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {static_cast<size_t>(max_pos * kRopeDim)}, /*random=*/true);
  BufferMeta d_positions = CreateBuffer<int64_t>(MEMORY_GPU, {max_batch});
  BufferMeta d_output_q_fp8 = CreateBuffer<uint8_t>(MEMORY_GPU, {max_q_elem});
  BufferMeta d_output_scale = CreateBuffer<float>(MEMORY_GPU, {max_w_elem});

  std::cout << std::setw(10) << "Batch" << std::setw(14) << "Time(ms)" << std::setw(14) << "BW(GB/s)" << std::endl;
  std::cout << std::string(38, '-') << std::endl;

  for (uint32_t batch_size : batch_sizes) {
    // Fill positions
    std::vector<int64_t> h_pos(batch_size);
    for (uint32_t i = 0; i < batch_size; ++i) h_pos[i] = static_cast<int64_t>(i % max_pos);
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_positions.data_ptr, h_pos.data(), batch_size * sizeof(int64_t), cudaMemcpyHostToDevice));

    const int warmups = 5;
    const int iterations = 20;

    auto fused_run = [&]() {
      fused_rope_quant_scale(d_q.data_ptr, d_weights.data_ptr, d_freqs.data_ptr, d_positions.data_ptr,
                             d_output_q_fp8.data_ptr, static_cast<float*>(d_output_scale.data_ptr), n_heads_inv_sqrt,
                             softmax_scale, num_heads, batch_size,
                             /*rope_offset=*/static_cast<uint32_t>(kNopeDim), stream);
    };
    float fused_ms = MeasureCudaExecutionTime(fused_run, stream, warmups, iterations);

    // Bandwidth calculation
    double input_bytes = static_cast<double>(batch_size) * num_heads * kHeadDim * sizeof(__nv_bfloat16);
    double weight_bytes = static_cast<double>(batch_size) * num_heads * sizeof(float);
    double freq_bytes = static_cast<double>(batch_size) * kRopeDim * sizeof(__nv_bfloat16);
    double output_bytes = static_cast<double>(batch_size) * num_heads * (kHeadDim + sizeof(float));
    double total_bytes = input_bytes + weight_bytes + freq_bytes + output_bytes;
    double bandwidth_gbps = total_bytes / (fused_ms * 1e-3) / 1e9;

    std::cout << std::setw(10) << batch_size << std::setw(14) << std::fixed << std::setprecision(4) << fused_ms
              << std::setw(14) << std::setprecision(1) << bandwidth_gbps << std::endl;
  }

  DeleteBuffer(d_q);
  DeleteBuffer(d_weights);
  DeleteBuffer(d_freqs);
  DeleteBuffer(d_positions);
  DeleteBuffer(d_output_q_fp8);
  DeleteBuffer(d_output_scale);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
