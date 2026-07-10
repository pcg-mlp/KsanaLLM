/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/others/vllm/main/deepseek_v4/fused_deepseek_v4_knorm_rope_kv_insert_kernel.h"

#include <cuda_fp8.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

// ─────────────────────────────────────────────────────────────────────────────
// Constants matching the kernel
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int kHeadDim = 512;
static constexpr int kRopeDim = 64;
static constexpr int kNopeDim = 448;
static constexpr int kEmbedDim = 32;
static constexpr int kNumQuantBlocks = 7;
static constexpr int kDsMlaTokenBytes = 576;  // 448 FP8 + 128 bf16
static constexpr int kE4M3TokenBytes = 512;   // dense FP8
static constexpr float FP8_E4M3_MAX = 448.0f;

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference
// ─────────────────────────────────────────────────────────────────────────────
static int32_t cpu_cast_to_ue8m0(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof(u));
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF);
  uint32_t mant = u & 0x7FFFFF;
  return exp + (mant != 0);
}

static float cpu_inv_scale_ue8m0(int32_t exp) {
  uint32_t u = static_cast<uint32_t>((127 + 127 - exp) << 23);
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

// Weighted RMSNorm + GPT-J RoPE on dims [448, 512).
// Returns elements in `out` (full 512 dims, post-norm + post-rope).
static void cpu_k_norm_rope(const float* k_in,           // [kHeadDim]
                            const float* norm_weight,    // [kHeadDim]
                            const float* cos_sin_cache,  // [kRopeDim] for this position
                            float eps, float* out) {     // [kHeadDim]
  float sum_sq = 0.0f;
  for (int i = 0; i < kHeadDim; ++i) {
    sum_sq += k_in[i] * k_in[i];
  }
  float rms_rcp = 1.0f / std::sqrt(sum_sq / static_cast<float>(kHeadDim) + eps);
  for (int i = 0; i < kHeadDim; ++i) {
    out[i] = k_in[i] * rms_rcp * norm_weight[i];
  }

  for (int i = kNopeDim; i < kHeadDim; i += 2) {
    int half_idx = (i - kNopeDim) / 2;
    float cos_val = cos_sin_cache[half_idx];
    float sin_val = cos_sin_cache[kEmbedDim + half_idx];
    float x = out[i];
    float y = out[i + 1];
    out[i] = x * cos_val - y * sin_val;
    out[i + 1] = x * sin_val + y * cos_val;
  }
}

// kFp8DsMla layout: 448 FP8 (per-64 UE8M0) for nope + 128 bytes bf16 for rope, plus 7 scales.
static void cpu_quantize_dsmla(const float* elements,  // [kHeadDim]
                               uint8_t* fp8_bytes,     // [448]
                               float* rope_out,        // [64]
                               uint8_t* scales) {      // [7]
  for (int i = 0; i < kRopeDim; ++i) {
    rope_out[i] = elements[kNopeDim + i];
  }
  for (int g = 0; g < kNumQuantBlocks; ++g) {
    int base = g * 64;
    float group_max = 0.0f;
    for (int i = 0; i < 64; ++i) {
      group_max = std::fmax(group_max, std::fabs(elements[base + i]));
    }
    float scale_raw = std::fmax(1e-4f, group_max) / FP8_E4M3_MAX;
    int32_t scale_ue8m0 = cpu_cast_to_ue8m0(scale_raw);
    float inv_scale = cpu_inv_scale_ue8m0(scale_ue8m0);
    scales[g] = static_cast<uint8_t>(scale_ue8m0);
    for (int i = 0; i < 64; ++i) {
      float scaled = elements[base + i] * inv_scale;
      scaled = std::fmin(std::fmax(scaled, -FP8_E4M3_MAX), FP8_E4M3_MAX);
      fp8_bytes[base + i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
    }
  }
}

// kFp8E4M3 layout: dense 512 FP8, no scales.
static void cpu_quantize_e4m3(const float* elements, uint8_t* fp8_bytes) {
  for (int i = 0; i < kHeadDim; ++i) {
    float scaled = std::fmin(std::fmax(elements[i], -FP8_E4M3_MAX), FP8_E4M3_MAX);
    fp8_bytes[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test fixture
// ─────────────────────────────────────────────────────────────────────────────
class FusedKNormRopeKVInsertTest : public NvidiaTestSuitBase {
 protected:
  using NvidiaTestSuitBase::stream;

  static constexpr int kPageBits = 1;
  static constexpr int kPageSize = 1 << kPageBits;

  template <llm_kernels::utils::KVCacheType KV_DTYPE>
  void TestCorrectness(int num_tokens) {
    constexpr int kTokenDataBytes =
        (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) ? kE4M3TokenBytes : kDsMlaTokenBytes;
    constexpr bool kHasScales = (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8DsMla);
    constexpr int kScaleStride = kHasScales ? 8 : 0;
    const int64_t page_stride =
        static_cast<int64_t>(kTokenDataBytes) * kPageSize + static_cast<int64_t>(kScaleStride) * kPageSize;

    const float eps = 1e-6f;
    const int max_pos = 128;

    // ── Prepare host data ──
    auto make_bf16_and_ref = [](size_t n, std::vector<__nv_bfloat16>& bf16_out, std::vector<float>& ref_out) {
      bf16_out.resize(n);
      ref_out.resize(n);
      for (size_t i = 0; i < n; ++i) {
        float v = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
        bf16_out[i] = __float2bfloat16(v);
        ref_out[i] = __bfloat162float(bf16_out[i]);
      }
    };

    srand(42);
    std::vector<__nv_bfloat16> h_kv_bf16;
    std::vector<float> h_kv_ref;
    make_bf16_and_ref(num_tokens * kHeadDim, h_kv_bf16, h_kv_ref);

    std::vector<__nv_bfloat16> h_weight_bf16;
    std::vector<float> h_weight_ref;
    make_bf16_and_ref(kHeadDim, h_weight_bf16, h_weight_ref);

    std::vector<__nv_bfloat16> h_cos_sin_bf16;
    std::vector<float> h_cos_sin_ref;
    make_bf16_and_ref(max_pos * kRopeDim, h_cos_sin_bf16, h_cos_sin_ref);

    std::vector<int64_t> h_positions(num_tokens);
    std::vector<int32_t> h_raw_out_loc(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
      h_positions[i] = i % max_pos;
      h_raw_out_loc[i] = i;
    }

    // ── Allocate device memory ──
    int num_pages = (num_tokens + kPageSize - 1) / kPageSize;
    size_t cache_elems = static_cast<size_t>(num_pages) * page_stride;

    BufferMeta d_kv = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {h_kv_bf16.size()});
    BufferMeta d_weight = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {h_weight_bf16.size()});
    BufferMeta d_cos_sin = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {h_cos_sin_bf16.size()});
    BufferMeta d_positions = CreateBuffer<int64_t>(MEMORY_GPU, {static_cast<size_t>(num_tokens)});
    BufferMeta d_raw_out_loc = CreateBuffer<int32_t>(MEMORY_GPU, {static_cast<size_t>(num_tokens)});
    BufferMeta d_cache = CreateBuffer<uint8_t>(MEMORY_GPU, {cache_elems});

    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_kv.data_ptr, h_kv_bf16.data(), d_kv.buf_size, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_weight.data_ptr, h_weight_bf16.data(), d_weight.buf_size, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_cos_sin.data_ptr, h_cos_sin_bf16.data(), d_cos_sin.buf_size, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_positions.data_ptr, h_positions.data(), d_positions.buf_size, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_raw_out_loc.data_ptr, h_raw_out_loc.data(), d_raw_out_loc.buf_size, cudaMemcpyHostToDevice));

    // ── Launch kernel ──
    InvokeFusedKNormRopeKVInsert<KV_DTYPE>(
        reinterpret_cast<__nv_bfloat16*>(d_kv.data_ptr), reinterpret_cast<__nv_bfloat16*>(d_weight.data_ptr),
        d_cache.data_ptr, reinterpret_cast<int64_t*>(d_positions.data_ptr),
        reinterpret_cast<__nv_bfloat16*>(d_cos_sin.data_ptr), reinterpret_cast<int32_t*>(d_raw_out_loc.data_ptr), eps,
        num_tokens, page_stride, kPageBits, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // ── Copy results back ──
    std::vector<uint8_t> h_cache(cache_elems);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_cache.data(), d_cache.data_ptr, d_cache.buf_size, cudaMemcpyDeviceToHost));

    // ── Verify ──
    for (int t = 0; t < num_tokens; ++t) {
      const float* kv_in = &h_kv_ref[t * kHeadDim];
      const float* cos_sin = &h_cos_sin_ref[h_positions[t] * kRopeDim];

      float expected_elems[kHeadDim];
      cpu_k_norm_rope(kv_in, h_weight_ref.data(), cos_sin, eps, expected_elems);

      int32_t index = h_raw_out_loc[t];
      int32_t page = index >> kPageBits;
      int32_t offset = index & ((1 << kPageBits) - 1);
      const uint8_t* page_ptr = &h_cache[page * page_stride];
      const uint8_t* value_ptr = page_ptr + offset * kTokenDataBytes;

      if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
        uint8_t expected_fp8[kHeadDim];
        cpu_quantize_e4m3(expected_elems, expected_fp8);
        for (int i = 0; i < kHeadDim; ++i) {
          EXPECT_EQ(value_ptr[i], expected_fp8[i]) << "E4M3 FP8 mismatch at token=" << t << " dim=" << i;
        }
      } else {
        uint8_t expected_fp8[kNopeDim];
        float expected_rope[kRopeDim];
        uint8_t expected_scales[kNumQuantBlocks];
        cpu_quantize_dsmla(expected_elems, expected_fp8, expected_rope, expected_scales);

        for (int i = 0; i < kNopeDim; ++i) {
          EXPECT_EQ(value_ptr[i], expected_fp8[i]) << "DsMla FP8 mismatch at token=" << t << " dim=" << i;
        }

        const __nv_bfloat16* rope_bf16_ptr = reinterpret_cast<const __nv_bfloat16*>(value_ptr + kNopeDim);
        for (int i = 0; i < kRopeDim; ++i) {
          float actual_f = __bfloat162float(rope_bf16_ptr[i]);
          EXPECT_NEAR(actual_f, expected_rope[i], 5e-3f) << "DsMla RoPE mismatch at token=" << t << " dim=" << i;
        }

        const uint8_t* scale_ptr = page_ptr + static_cast<int64_t>(kTokenDataBytes) * kPageSize + offset * 8;
        for (int g = 0; g < kNumQuantBlocks; ++g) {
          EXPECT_EQ(scale_ptr[g], expected_scales[g]) << "DsMla scale mismatch at token=" << t << " group=" << g;
        }
      }
    }

    // ── Cleanup ──
    DeleteBuffer(d_kv);
    DeleteBuffer(d_weight);
    DeleteBuffer(d_cache);
    DeleteBuffer(d_positions);
    DeleteBuffer(d_cos_sin);
    DeleteBuffer(d_raw_out_loc);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Correctness tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(FusedKNormRopeKVInsertTest, DsMlaSingleToken) {
  TestCorrectness<llm_kernels::utils::KVCacheType::kFp8DsMla>(/*num_tokens=*/1);
}

TEST_F(FusedKNormRopeKVInsertTest, DsMlaMultiToken) {
  TestCorrectness<llm_kernels::utils::KVCacheType::kFp8DsMla>(/*num_tokens=*/8);
}

TEST_F(FusedKNormRopeKVInsertTest, E4M3SingleToken) {
  TestCorrectness<llm_kernels::utils::KVCacheType::kFp8E4M3>(/*num_tokens=*/1);
}

TEST_F(FusedKNormRopeKVInsertTest, E4M3MultiToken) {
  TestCorrectness<llm_kernels::utils::KVCacheType::kFp8E4M3>(/*num_tokens=*/8);
}

// ─────────────────────────────────────────────────────────────────────────────
// Performance test (disabled by default)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(FusedKNormRopeKVInsertTest, DISABLED_PerfTest) {
  const std::vector<int> token_nums = {1, 8, 32, 128, 512, 2048};
  const float eps = 1e-6f;
  const int max_pos = 8192;
  constexpr int page_bits = 6;
  constexpr int page_size = 1 << page_bits;
  constexpr int64_t page_stride = static_cast<int64_t>(kDsMlaTokenBytes) * page_size + 8 * page_size;

  for (int num_tokens : token_nums) {
    size_t kv_elems = num_tokens * kHeadDim;
    int num_pages = (num_tokens + page_size - 1) / page_size;

    BufferMeta d_kv = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {kv_elems}, true);
    BufferMeta d_weight = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {static_cast<size_t>(kHeadDim)}, true);
    BufferMeta d_cache = CreateBuffer<uint8_t>(MEMORY_GPU, {static_cast<size_t>(num_pages * page_stride)}, false);
    BufferMeta d_positions = CreateBuffer<int64_t>(MEMORY_GPU, {static_cast<size_t>(num_tokens)}, false);
    BufferMeta d_cos_sin = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {static_cast<size_t>(max_pos * kRopeDim)}, true);
    BufferMeta d_raw_out_loc = CreateBuffer<int32_t>(MEMORY_GPU, {static_cast<size_t>(num_tokens)}, false);

    std::vector<int64_t> h_pos(num_tokens);
    std::vector<int32_t> h_loc(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
      h_pos[i] = i;
      h_loc[i] = i;
    }
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_positions.data_ptr, h_pos.data(), num_tokens * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_raw_out_loc.data_ptr, h_loc.data(), num_tokens * sizeof(int32_t), cudaMemcpyHostToDevice));

    const int warmups = 5;
    const int iterations = 20;
    auto cuda_run = [&]() {
      InvokeFusedKNormRopeKVInsert<llm_kernels::utils::KVCacheType::kFp8DsMla>(
          reinterpret_cast<__nv_bfloat16*>(d_kv.data_ptr), reinterpret_cast<__nv_bfloat16*>(d_weight.data_ptr),
          d_cache.data_ptr, reinterpret_cast<int64_t*>(d_positions.data_ptr),
          reinterpret_cast<__nv_bfloat16*>(d_cos_sin.data_ptr), reinterpret_cast<int32_t*>(d_raw_out_loc.data_ptr), eps,
          num_tokens, page_stride, page_bits, stream);
    };
    float elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, warmups, iterations);
    std::cout << "Tokens: " << num_tokens << ", Time: " << elapsed_ms << " ms" << std::endl;

    DeleteBuffer(d_kv);
    DeleteBuffer(d_weight);
    DeleteBuffer(d_cache);
    DeleteBuffer(d_positions);
    DeleteBuffer(d_cos_sin);
    DeleteBuffer(d_raw_out_loc);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
