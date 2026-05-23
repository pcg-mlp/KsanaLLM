/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/store.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {


static constexpr int kHeadDim = 128;
static constexpr int kScaleBytes = 4;
static constexpr float FP8_E4M3_MAX = 448.0f;

namespace {

void cpu_store_one_token(const float* input, uint8_t* fp8_out, float* scale_out) {
  // Per-warp abs-max reduction to determine quantization scale
  float abs_max = 0.0f;
  for (int i = 0; i < kHeadDim; ++i) {
    abs_max = std::fmax(abs_max, std::fabs(input[i]));
  }
  float scale = std::fmax(1e-4f, abs_max) / FP8_E4M3_MAX;
  float inv_scale = 1.0f / scale;
  *scale_out = scale;

  for (int i = 0; i < kHeadDim; ++i) {
    float val = input[i] * inv_scale;
    val = std::fmin(std::fmax(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
    fp8_out[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3));
  }
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Test fixture
// ─────────────────────────────────────────────────────────────────────────────
class FusedGatherIndexerCacheTest : public NvidiaTestSuitBase {
 protected:
  using NvidiaTestSuitBase::stream;

  static int64_t PageStride(uint32_t page_bits) {
    // 128 bytes FP8 + 4 bytes scale = 132 bytes per token per page
    return static_cast<int64_t>(kHeadDim + kScaleBytes) * (1u << page_bits);
  }

  // Builds sequential block_table and linear seq_len_offset for gather tests.
  struct GatherSetup {
    std::vector<size_t> seq_len_offset;
    std::vector<int32_t> block_table;
    int32_t max_num_blocks_per_seq;
  };

  static GatherSetup BuildGatherSetup(uint32_t num_tokens, uint32_t batch_size, uint32_t page_size,
                                      int32_t compress_ratio) {
    GatherSetup s;
    uint32_t tokens_per_batch = num_tokens / batch_size;
    uint32_t blocks_per_batch = (tokens_per_batch + page_size - 1) / page_size;
    s.max_num_blocks_per_seq = static_cast<int32_t>(blocks_per_batch);

    // seq_len_offset is in uncompressed units
    s.seq_len_offset.resize(batch_size + 1);
    for (uint32_t b = 0; b <= batch_size; ++b) {
      s.seq_len_offset[b] = static_cast<size_t>(b) * tokens_per_batch * compress_ratio;
    }

    s.block_table.assign(batch_size * s.max_num_blocks_per_seq, -1);
    for (uint32_t b = 0; b < batch_size; ++b) {
      for (uint32_t blk = 0; blk < blocks_per_batch; ++blk) {
        s.block_table[b * s.max_num_blocks_per_seq + blk] = static_cast<int32_t>(b * blocks_per_batch + blk);
      }
    }
    return s;
  }

  // End-to-end correctness test: store -> gather -> verify.
  template <uint32_t kPageBits>
  void TestStoreGatherRoundTrip(uint32_t num_tokens, uint32_t batch_size, int32_t compress_ratio) {
    ASSERT_EQ(num_tokens % batch_size, 0u) << "num_tokens must be divisible by batch_size";

    constexpr uint32_t page_bits = kPageBits;
    const uint32_t page_size = 1u << page_bits;
    const int64_t page_stride = PageStride(page_bits);

    // Generate random bf16 input and its exact float representation
    std::mt19937 rng(12345u);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    const size_t in_elem = static_cast<size_t>(num_tokens) * kHeadDim;

    std::vector<__nv_bfloat16> in_bf16(in_elem);
    std::vector<float> in_f(in_elem);
    for (size_t i = 0; i < in_elem; ++i) {
      in_bf16[i] = __float2bfloat16(dist(rng));
      in_f[i] = __bfloat162float(in_bf16[i]);
    }

    // Sequential store indices (identity mapping)
    uint32_t num_pages = (num_tokens + page_size - 1) / page_size;
    std::vector<int32_t> store_indices(num_tokens);
    std::iota(store_indices.begin(), store_indices.end(), 0);

    auto setup = BuildGatherSetup(num_tokens, batch_size, page_size, compress_ratio);

    // Allocate device memory
    size_t cache_bytes = static_cast<size_t>(num_pages) * page_stride;
    BufferMeta d_input = CreateBuffer<__nv_bfloat16>(MEMORY_GPU, {in_elem});
    BufferMeta d_store_indices = CreateBuffer<int32_t>(MEMORY_GPU, {num_tokens});
    BufferMeta d_cache = CreateBuffer<uint8_t>(MEMORY_GPU, {cache_bytes});
    BufferMeta d_block_table = CreateBuffer<int32_t>(MEMORY_GPU, {setup.block_table.size()});
    BufferMeta d_seq_len_offset = CreateBuffer<size_t>(MEMORY_GPU, {setup.seq_len_offset.size()});
    BufferMeta d_k_fp8 = CreateBuffer<uint8_t>(MEMORY_GPU, {static_cast<size_t>(num_tokens) * kHeadDim});
    BufferMeta d_k_scale = CreateBuffer<float>(MEMORY_GPU, {num_tokens});

    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_input.data_ptr, in_bf16.data(), in_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_store_indices.data_ptr, store_indices.data(), num_tokens * sizeof(int32_t),
                                       cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_block_table.data_ptr, setup.block_table.data(),
                                       setup.block_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_seq_len_offset.data_ptr, setup.seq_len_offset.data(),
                                       setup.seq_len_offset.size() * sizeof(size_t), cudaMemcpyHostToDevice));

    // Store
    fused_store_indexer_cache<__nv_bfloat16, int32_t, kPageBits>(
        d_input.data_ptr, d_cache.data_ptr, d_store_indices.data_ptr, num_tokens, page_stride, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Gather
    fused_gather_indexer_cache<__nv_bfloat16, int32_t, kPageBits>(
        d_k_fp8.data_ptr, d_k_scale.data_ptr, d_cache.data_ptr, d_block_table.data_ptr,
        reinterpret_cast<size_t*>(d_seq_len_offset.data_ptr), batch_size, num_tokens, page_stride,
        setup.max_num_blocks_per_seq, compress_ratio, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Verify against CPU reference
    std::vector<uint8_t> h_k_fp8(static_cast<size_t>(num_tokens) * kHeadDim);
    std::vector<float> h_k_scale(num_tokens);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_k_fp8.data(), d_k_fp8.data_ptr, h_k_fp8.size(), cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(h_k_scale.data(), d_k_scale.data_ptr, num_tokens * sizeof(float), cudaMemcpyDeviceToHost));

    for (uint32_t t = 0; t < num_tokens; ++t) {
      uint8_t expected_fp8[kHeadDim];
      float expected_scale;
      cpu_store_one_token(&in_f[t * kHeadDim], expected_fp8, &expected_scale);

      ASSERT_FLOAT_EQ(h_k_scale[t], expected_scale) << "Scale mismatch at token=" << t;

      // Fast-path: memcmp whole row, only itemize on mismatch
      if (std::memcmp(&h_k_fp8[t * kHeadDim], expected_fp8, kHeadDim) != 0) {
        for (int i = 0; i < kHeadDim; ++i) {
          ASSERT_EQ(h_k_fp8[t * kHeadDim + i], expected_fp8[i]) << "FP8 mismatch at token=" << t << " dim=" << i;
        }
      }
    }

    DeleteBuffer(d_input);
    DeleteBuffer(d_store_indices);
    DeleteBuffer(d_cache);
    DeleteBuffer(d_block_table);
    DeleteBuffer(d_seq_len_offset);
    DeleteBuffer(d_k_fp8);
    DeleteBuffer(d_k_scale);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Correctness tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(FusedGatherIndexerCacheTest, SingleBatchSinglePage) {
  TestStoreGatherRoundTrip<6>(/*num_tokens=*/64, /*batch_size=*/1, /*compress_ratio=*/1);
}

TEST_F(FusedGatherIndexerCacheTest, SingleBatchMultiPage) {
  TestStoreGatherRoundTrip<6>(/*num_tokens=*/256, /*batch_size=*/1, /*compress_ratio=*/1);
}

TEST_F(FusedGatherIndexerCacheTest, MultiBatch) {
  TestStoreGatherRoundTrip<6>(/*num_tokens=*/256, /*batch_size=*/4, /*compress_ratio=*/1);
}

TEST_F(FusedGatherIndexerCacheTest, CompressRatio4) {
  TestStoreGatherRoundTrip<6>(/*num_tokens=*/128, /*batch_size=*/2, /*compress_ratio=*/4);
}

TEST_F(FusedGatherIndexerCacheTest, PageBits1) {
  TestStoreGatherRoundTrip<1>(/*num_tokens=*/4, /*batch_size=*/2, /*compress_ratio=*/1);
}

TEST_F(FusedGatherIndexerCacheTest, LargeScale) {
  TestStoreGatherRoundTrip<6>(/*num_tokens=*/2048, /*batch_size=*/8, /*compress_ratio=*/4);
}

// ─────────────────────────────────────────────────────────────────────────────
// Performance test
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(FusedGatherIndexerCacheTest, DISABLED_PerfTest) {
  const std::vector<uint32_t> token_nums = {64, 256, 1024, 4096, 16384};
  constexpr uint32_t page_bits = 6;
  constexpr uint32_t page_size = 1u << page_bits;
  const int64_t page_stride = PageStride(page_bits);
  constexpr int32_t compress_ratio = 4;
  constexpr uint32_t batch_size = 8;

  std::cout << "\n=== fused_gather_indexer_cache Performance ===\n";
  std::cout << "HeadDim=" << kHeadDim << ", PageBits=" << page_bits << ", CompressRatio=" << compress_ratio
            << ", BatchSize=" << batch_size << "\n\n";

  for (uint32_t num_tokens : token_nums) {
    uint32_t tokens_per_batch = num_tokens / batch_size;
    uint32_t blocks_per_batch = (tokens_per_batch + page_size - 1) / page_size;
    uint32_t num_pages = blocks_per_batch * batch_size;

    auto setup = BuildGatherSetup(num_tokens, batch_size, page_size, compress_ratio);

    size_t cache_bytes = static_cast<size_t>(num_pages) * page_stride;
    BufferMeta d_cache = CreateBuffer<uint8_t>(MEMORY_GPU, {cache_bytes}, /*random=*/true);
    BufferMeta d_block_table = CreateBuffer<int32_t>(MEMORY_GPU, {setup.block_table.size()});
    BufferMeta d_seq_len_offset = CreateBuffer<size_t>(MEMORY_GPU, {setup.seq_len_offset.size()});
    BufferMeta d_k_fp8 = CreateBuffer<uint8_t>(MEMORY_GPU, {static_cast<size_t>(num_tokens) * kHeadDim});
    BufferMeta d_k_scale = CreateBuffer<float>(MEMORY_GPU, {num_tokens});

    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_block_table.data_ptr, setup.block_table.data(),
                                       setup.block_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_seq_len_offset.data_ptr, setup.seq_len_offset.data(),
                                       setup.seq_len_offset.size() * sizeof(size_t), cudaMemcpyHostToDevice));

    auto cuda_run = [&]() {
      fused_gather_indexer_cache<__nv_bfloat16, int32_t, 6>(
          d_k_fp8.data_ptr, d_k_scale.data_ptr, d_cache.data_ptr, d_block_table.data_ptr,
          reinterpret_cast<size_t*>(d_seq_len_offset.data_ptr), batch_size, num_tokens, page_stride,
          setup.max_num_blocks_per_seq, compress_ratio, stream);
    };
    float elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, 10, 100);

    double bytes_per_token = (kHeadDim + kScaleBytes) * 2.0;  // read + write
    double total_bytes = bytes_per_token * num_tokens;
    double bandwidth_gbps = total_bytes / (elapsed_ms * 1e-3) / 1e9;

    std::cout << "  Tokens: " << std::setw(6) << num_tokens << ", Time: " << std::fixed << std::setprecision(4)
              << elapsed_ms << " ms"
              << ", BW: " << std::setprecision(1) << bandwidth_gbps << " GB/s" << std::endl;

    DeleteBuffer(d_cache);
    DeleteBuffer(d_block_table);
    DeleteBuffer(d_seq_len_offset);
    DeleteBuffer(d_k_fp8);
    DeleteBuffer(d_k_scale);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
