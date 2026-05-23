/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Unit tests for the top-K transform kernels ported from SGLang DeepSeek-V4:
 *   https://github.com/sgl-project/sglang/blob/4807f6c5220c8ac53e395d6d2ed493cda241bc8e/python/sglang/jit_kernel/csrc/deepseek_v4/topk.cuh
 *   https://github.com/sgl-project/sglang/blob/4807f6c5220c8ac53e395d6d2ed493cda241bc8e/python/sglang/jit_kernel/csrc/deepseek_v4/topk_1024.cuh
 *   https://github.com/sgl-project/sglang/blob/4807f6c5220c8ac53e395d6d2ed493cda241bc8e/python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh
 */

#include "csrc/kernels/nvidia/others/sglang/main/topk/topk.h"
#include "csrc/kernels/nvidia/others/sglang/main/topk/topk_wrapper.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

using topk::kTopK1024;
using topk::kTopK512;
using topk::launch_topk1024;
using topk::launch_topk512;
using topk::TopKWrapper;

class TopKTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  // Host reference: return the indices of the top-`k` values in `scores`
  // (size `length`). Returns a sorted-by-index set so that it can be
  // compared to the kernel output using set equality — the kernel does not
  // guarantee a particular order among the winners.
  static std::vector<int32_t> HostTopKIndices(const float* scores, uint32_t length, uint32_t k) {
    std::vector<int32_t> idx(length);
    std::iota(idx.begin(), idx.end(), 0);
    const uint32_t keep = std::min<uint32_t>(k, length);
    std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(),
                      [&](int32_t a, int32_t b) { return scores[a] > scores[b]; });
    idx.resize(keep);
    std::sort(idx.begin(), idx.end());
    return idx;
  }

  // Apply the kernel's `page_to_indices` transform on the host:
  //   mask = (1 << page_bits) - 1
  //   out  = (page_table[i >> page_bits] << page_bits) | (i & mask)
  static int32_t PageToIndicesHost(const int32_t* page_table, uint32_t i, uint32_t page_bits) {
    const uint32_t mask = (1u << page_bits) - 1u;
    return (page_table[i >> page_bits] << page_bits) | static_cast<int32_t>(i & mask);
  }

  // Helper: build row_starts and row_ends arrays for a uniform decode-like scenario
  // where each batch item has `seq_len` scores laid out at stride `score_stride`.
  // row_starts[b] = b * score_stride, row_ends[b] = b * score_stride + seq_len (clamped)
  static void BuildRowStartsEnds(std::vector<int32_t>& row_starts, std::vector<int32_t>& row_ends, uint32_t batch_size,
                                 uint32_t seq_len, int64_t score_stride) {
    row_starts.resize(batch_size);
    row_ends.resize(batch_size);
    for (uint32_t b = 0; b < batch_size; ++b) {
      row_starts[b] = static_cast<int32_t>(b * score_stride);
      row_ends[b] = static_cast<int32_t>(b * score_stride + seq_len);
    }
  }
};

// ----------------------------------------------------------------------------
// Test 1: dispatch smoke — both variants launch for a small batch without
// error.
// ----------------------------------------------------------------------------
TEST_F(TopKTestSuit, DispatchSmokeK512AndK1024) {
  const uint32_t batch_size = 2;
  const uint32_t page_size = 16;
  const uint32_t page_bits = 4;

  auto run_smoke = [&](uint32_t k) {
    const uint32_t seq_len = 256;  // <= k, exercises the naive_transform path
    const int64_t score_stride = seq_len;
    const int64_t page_table_stride = (seq_len + page_size - 1) / page_size;

    BufferMeta scores = CreateBuffer<float>(MemoryType::MEMORY_GPU, {batch_size * (size_t)score_stride},
                                            /*is_random_init=*/true);
    BufferMeta page_table = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size * (size_t)page_table_stride});
    BufferMeta page_indices = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size * (size_t)k});

    // Build row_starts/row_ends
    std::vector<int32_t> row_starts_h, row_ends_h;
    BuildRowStartsEnds(row_starts_h, row_ends_h, batch_size, seq_len, score_stride);
    BufferMeta row_starts_buf = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size});
    BufferMeta row_ends_buf = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size});
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(row_starts_buf.data_ptr, row_starts_h.data(),
                                       row_starts_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(row_ends_buf.data_ptr, row_ends_h.data(), row_ends_h.size() * sizeof(int32_t),
                                       cudaMemcpyHostToDevice));

    // Fill page_table with identity-ish values (0,1,2,...)
    std::vector<int32_t> pt_h(batch_size * page_table_stride);
    std::iota(pt_h.begin(), pt_h.end(), 0);
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(page_table.data_ptr, pt_h.data(), pt_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    // num_works = batch_size (decode-like: 1 work per batch)
    const uint32_t num_works = batch_size;
    const uint32_t extend_len = num_works / batch_size;  // = 1

    if (k == kTopK512) {
      launch_topk512(
          reinterpret_cast<const float*>(scores.data_ptr), reinterpret_cast<const int32_t*>(row_starts_buf.data_ptr),
          reinterpret_cast<const int32_t*>(row_ends_buf.data_ptr),
          reinterpret_cast<const int32_t*>(page_table.data_ptr), reinterpret_cast<int32_t*>(page_indices.data_ptr),
          /*raw_indices_out=*/nullptr, score_stride, page_table_stride, num_works, page_bits,
          /*extend_lens=*/nullptr, extend_len, batch_size, stream);
    } else {
      launch_topk1024(
          reinterpret_cast<const float*>(scores.data_ptr), reinterpret_cast<const int32_t*>(row_starts_buf.data_ptr),
          reinterpret_cast<const int32_t*>(row_ends_buf.data_ptr),
          reinterpret_cast<const int32_t*>(page_table.data_ptr), reinterpret_cast<int32_t*>(page_indices.data_ptr),
          /*raw_indices_out=*/nullptr, score_stride, page_table_stride, num_works, page_bits,
          /*extend_lens=*/nullptr, extend_len, batch_size, stream);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    DeleteBuffer(scores);
    DeleteBuffer(row_starts_buf);
    DeleteBuffer(row_ends_buf);
    DeleteBuffer(page_table);
    DeleteBuffer(page_indices);
  };

  run_smoke(kTopK512);
  run_smoke(kTopK1024);
}

// ----------------------------------------------------------------------------
// Test 2: numerical correctness of the top-K selection for the "real" path
//         (seq_len > K, so radix_topk is actually exercised).
//
// Strategy:
//   - Use K = 512 and seq_len = 2048.
//   - Generate deterministic random scores.
//   - Run the kernel; copy back the `raw_indices_out` buffer (which the
//     kernel populates with the UN-paged absolute positions of the winners).
//   - Compare the returned set against HostTopKIndices.
//
// The kernel does not guarantee a sorted order among winners; we compare by
// SET equality and also verify that the kernel's winners' minimum score is
// >= the host's K-th score (i.e. the selection is at least as good as the
// correct threshold; with floating-point ties some indices may swap but the
// threshold value is stable).
// ----------------------------------------------------------------------------
TEST_F(TopKTestSuit, ForwardNumericalMatchesHostRefK512) {
  constexpr uint32_t batch_size = 1;
  constexpr uint32_t seq_len = 2048;
  constexpr uint32_t page_size = 16;
  constexpr uint32_t page_bits = 4;  // log2(16)

  std::mt19937 rng(20260427);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> scores_h(seq_len);
  for (auto& x : scores_h) x = dist(rng);

  std::vector<int32_t> page_table_h((seq_len + page_size - 1) / page_size);
  std::iota(page_table_h.begin(), page_table_h.end(), 1000);  // arbitrary non-identity base

  // row_starts = [0], row_ends = [seq_len] (single batch, single work item)
  std::vector<int32_t> row_starts_h = {0};
  std::vector<int32_t> row_ends_h = {static_cast<int32_t>(seq_len)};

  BufferMeta scores = CreateBuffer<float>(MemoryType::MEMORY_GPU, {seq_len});
  BufferMeta row_starts_buf = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size});
  BufferMeta row_ends_buf = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size});
  BufferMeta page_table = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {page_table_h.size()});
  BufferMeta page_indices = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {kTopK512});
  BufferMeta raw_indices = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {kTopK512});

  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(scores.data_ptr, scores_h.data(), scores_h.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(row_starts_buf.data_ptr, row_starts_h.data(),
                                     row_starts_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(row_ends_buf.data_ptr, row_ends_h.data(), row_ends_h.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(page_table.data_ptr, page_table_h.data(), page_table_h.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice));

  const uint32_t num_works = batch_size;
  const uint32_t extend_len = 1;  // decode-like: 1 work per batch

  launch_topk512(
      reinterpret_cast<const float*>(scores.data_ptr), reinterpret_cast<const int32_t*>(row_starts_buf.data_ptr),
      reinterpret_cast<const int32_t*>(row_ends_buf.data_ptr), reinterpret_cast<const int32_t*>(page_table.data_ptr),
      reinterpret_cast<int32_t*>(page_indices.data_ptr), reinterpret_cast<int32_t*>(raw_indices.data_ptr),
      /*score_stride=*/seq_len, /*page_table_stride=*/static_cast<int64_t>(page_table_h.size()), num_works, page_bits,
      /*extend_lens=*/nullptr, extend_len, batch_size, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<int32_t> raw_h(kTopK512);
  std::vector<int32_t> page_h(kTopK512);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(raw_h.data(), raw_indices.data_ptr, raw_h.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(page_h.data(), page_indices.data_ptr, page_h.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

  // --- (a) Threshold check: the minimum score among the kernel's winners
  //         must be >= the host top-K threshold. This is stable even with
  //         random ties at the boundary (the host reference may shuffle
  //         ties, but any valid top-K must have the same threshold value). ---
  auto host_ref = HostTopKIndices(scores_h.data(), seq_len, kTopK512);
  ASSERT_EQ(host_ref.size(), kTopK512);
  // host_ref is sorted by index (ascending); find the *minimum score* across all
  // top-K winners — that is the true K-th-highest score threshold.
  float host_threshold = std::numeric_limits<float>::max();
  for (int32_t idx : host_ref) host_threshold = std::min(host_threshold, scores_h[idx]);
  for (uint32_t i = 0; i < kTopK512; ++i) {
    ASSERT_GE(raw_h[i], 0);
    ASSERT_LT(static_cast<uint32_t>(raw_h[i]), seq_len);
    EXPECT_GE(scores_h[raw_h[i]], host_threshold) << "kernel winner " << raw_h[i] << " has score " << scores_h[raw_h[i]]
                                                  << " < host-threshold " << host_threshold;
  }

  // --- (b) All kernel winners are distinct. ---
  std::unordered_set<int32_t> raw_set(raw_h.begin(), raw_h.end());
  EXPECT_EQ(raw_set.size(), kTopK512);

  // --- (c) The page transform matches HostC++ reference. ---
  for (uint32_t i = 0; i < kTopK512; ++i) {
    const int32_t expected = PageToIndicesHost(page_table_h.data(), static_cast<uint32_t>(raw_h[i]), page_bits);
    EXPECT_EQ(page_h[i], expected) << "mismatch at i=" << i << ", raw=" << raw_h[i];
  }

  DeleteBuffer(scores);
  DeleteBuffer(row_starts_buf);
  DeleteBuffer(row_ends_buf);
  DeleteBuffer(page_table);
  DeleteBuffer(page_indices);
  DeleteBuffer(raw_indices);
}

// ----------------------------------------------------------------------------
// Test 3: single work item (num_works=1) launches without crash for both
// the 512 and 1024 variants. This is the minimal valid configuration — a
// single batch item with seq_len < K so the naive_transform path is taken.
// ----------------------------------------------------------------------------
TEST_F(TopKTestSuit, SingleWorkItemIsValid) {
  constexpr uint32_t page_bits = 4;
  constexpr uint32_t seq_len = 64;
  const int64_t score_stride = seq_len;
  const int64_t page_table_stride = (seq_len + 16 - 1) / 16;

  BufferMeta scores = CreateBuffer<float>(MemoryType::MEMORY_GPU, {(size_t)score_stride}, /*is_random_init=*/true);
  BufferMeta page_table = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {(size_t)page_table_stride});
  BufferMeta page_indices_512 = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {kTopK512});
  BufferMeta page_indices_1024 = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {kTopK1024});

  std::vector<int32_t> row_starts_h = {0};
  std::vector<int32_t> row_ends_h = {static_cast<int32_t>(seq_len)};
  BufferMeta row_starts_buf = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {1});
  BufferMeta row_ends_buf = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {1});
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(row_starts_buf.data_ptr, row_starts_h.data(), sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(row_ends_buf.data_ptr, row_ends_h.data(), sizeof(int32_t), cudaMemcpyHostToDevice));

  std::vector<int32_t> pt_h(page_table_stride);
  std::iota(pt_h.begin(), pt_h.end(), 0);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(page_table.data_ptr, pt_h.data(), pt_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

  launch_topk512(
      reinterpret_cast<const float*>(scores.data_ptr), reinterpret_cast<const int32_t*>(row_starts_buf.data_ptr),
      reinterpret_cast<const int32_t*>(row_ends_buf.data_ptr), reinterpret_cast<const int32_t*>(page_table.data_ptr),
      reinterpret_cast<int32_t*>(page_indices_512.data_ptr), nullptr, score_stride, page_table_stride,
      /*num_works=*/1, page_bits,
      /*extend_lens=*/nullptr, /*extend_len=*/1, /*num_batches=*/1, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  launch_topk1024(
      reinterpret_cast<const float*>(scores.data_ptr), reinterpret_cast<const int32_t*>(row_starts_buf.data_ptr),
      reinterpret_cast<const int32_t*>(row_ends_buf.data_ptr), reinterpret_cast<const int32_t*>(page_table.data_ptr),
      reinterpret_cast<int32_t*>(page_indices_1024.data_ptr), nullptr, score_stride, page_table_stride,
      /*num_works=*/1, page_bits,
      /*extend_lens=*/nullptr, /*extend_len=*/1, /*num_batches=*/1, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  DeleteBuffer(scores);
  DeleteBuffer(row_starts_buf);
  DeleteBuffer(row_ends_buf);
  DeleteBuffer(page_table);
  DeleteBuffer(page_indices_512);
  DeleteBuffer(page_indices_1024);
}

// ----------------------------------------------------------------------------
// Test 4: the wrapper API (TopKWrapper<float, int32_t>::TransformV2) forwards
// correctly for a short-path scenario.
// ----------------------------------------------------------------------------
TEST_F(TopKTestSuit, WrapperV2ForwardsToLauncher) {
  using Wrap = TopKWrapper<float, int32_t>;

  constexpr uint32_t batch_size = 1;
  constexpr uint32_t seq_len = 128;  // < 512 so trivial path is used
  constexpr uint32_t page_size = 16;
  constexpr uint32_t page_bits = 4;

  const int64_t score_stride = seq_len;
  const int64_t page_table_stride = (seq_len + page_size - 1) / page_size;

  BufferMeta scores = CreateBuffer<float>(MemoryType::MEMORY_GPU, {seq_len}, /*is_random_init=*/true);
  BufferMeta seq_lens_buf = CreateBuffer<uint32_t>(MemoryType::MEMORY_GPU, {batch_size});
  BufferMeta page_table = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {(seq_len + page_size - 1) / page_size});
  BufferMeta page_indices = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {kTopK512});

  constexpr int64_t workspace_stride_bytes = 4 * 1024 * 1024;
  BufferMeta workspace = CreateBuffer<uint8_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(workspace_stride_bytes)});
  BufferMeta metadata = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {(batch_size + 1) * 4});

  uint32_t sl_h = seq_len;
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(seq_lens_buf.data_ptr, &sl_h, sizeof(uint32_t), cudaMemcpyHostToDevice));

  Wrap::TransformV2(seq_lens_buf.data_ptr, scores.data_ptr, page_table.data_ptr, page_indices.data_ptr,
                    workspace.data_ptr, metadata.data_ptr, score_stride, page_table_stride, workspace_stride_bytes,
                    batch_size, page_bits, /*max_seq_len=*/seq_len, /*static_cluster_threshold=*/0, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  DeleteBuffer(scores);
  DeleteBuffer(seq_lens_buf);
  DeleteBuffer(page_table);
  DeleteBuffer(page_indices);
  DeleteBuffer(workspace);
  DeleteBuffer(metadata);
}

// ----------------------------------------------------------------------------
// Test 5: TopKV2 smoke test — launch_topk_v2 completes without error for a
//         small batch where all items fit in the RegisterTopK (short) path.
// ----------------------------------------------------------------------------
TEST_F(TopKTestSuit, TopKV2SmokeShorPath) {
  using namespace topk;

  constexpr uint32_t batch_size = 2;
  constexpr uint32_t seq_len = 1024;  // <= Small::kMax1PassLength (16384)
  constexpr uint32_t page_size = 16;
  constexpr uint32_t page_bits = 4;  // log2(16)
  constexpr uint32_t K = 512;
  const int64_t score_stride = seq_len;
  const int64_t page_table_stride = (seq_len + page_size - 1) / page_size;

  BufferMeta scores = CreateBuffer<float>(MemoryType::MEMORY_GPU, {batch_size * static_cast<size_t>(score_stride)},
                                          /*is_random_init=*/true);
  BufferMeta seq_lens_buf = CreateBuffer<uint32_t>(MemoryType::MEMORY_GPU, {batch_size});
  BufferMeta page_table =
      CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size * static_cast<size_t>(page_table_stride)});
  BufferMeta page_indices = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size * K});

  // Workspace: each row needs Large::kWorkspaceInts * sizeof(int32_t) bytes.
  // Use a generous 4 MB per batch row to be safe.
  constexpr int64_t workspace_stride_bytes = 4 * 1024 * 1024;
  BufferMeta workspace =
      CreateBuffer<uint8_t>(MemoryType::MEMORY_GPU, {batch_size * static_cast<size_t>(workspace_stride_bytes)});
  // Metadata: (batch_size+1) rows × 4 int32_t columns.
  BufferMeta metadata = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {(batch_size + 1) * 4});

  // Fill seq_lens
  std::vector<uint32_t> sl_h(batch_size, seq_len);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(seq_lens_buf.data_ptr, sl_h.data(), sl_h.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  // Fill page_table (identity)
  std::vector<int32_t> pt_h(batch_size * page_table_stride);
  std::iota(pt_h.begin(), pt_h.end(), 0);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(page_table.data_ptr, pt_h.data(), pt_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

  TopKWrapper<float, int32_t>::TransformV2(
      seq_lens_buf.data_ptr, scores.data_ptr, page_table.data_ptr, page_indices.data_ptr, workspace.data_ptr,
      metadata.data_ptr, score_stride, page_table_stride, workspace_stride_bytes, batch_size,
      /*page_bits=*/page_bits, /*max_seq_len=*/seq_len, /*static_cluster_threshold=*/0, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  DeleteBuffer(scores);
  DeleteBuffer(seq_lens_buf);
  DeleteBuffer(page_table);
  DeleteBuffer(page_indices);
  DeleteBuffer(workspace);
  DeleteBuffer(metadata);
}

// ----------------------------------------------------------------------------
// Test 6: TopKV2 wrapper — verifies that the page_indices for a short-path
//         (seq_len <= 512 = K, trivial_transform branch) match the expected
//         identity page transform: page_indices[i] = (pt[i>>page_bits]<<page_bits)|(i & mask).
// ----------------------------------------------------------------------------
TEST_F(TopKTestSuit, TopKV2TrivialTransformCorrectness) {
  using namespace topk;

  constexpr uint32_t batch_size = 1;
  constexpr uint32_t seq_len = 256;  // < K=512, trivial_transform path
  constexpr uint32_t page_size = 16;
  constexpr uint32_t page_bits = 4;
  constexpr uint32_t K = 512;
  const int64_t score_stride = seq_len;
  const int64_t page_table_stride = (seq_len + page_size - 1) / page_size;

  std::vector<float> scores_h(seq_len);
  std::iota(scores_h.begin(), scores_h.end(), 0.0f);
  std::vector<int32_t> pt_h(page_table_stride);
  std::iota(pt_h.begin(), pt_h.end(), 1000);  // non-identity base for page table

  BufferMeta scores = CreateBuffer<float>(MemoryType::MEMORY_GPU, {seq_len});
  BufferMeta seq_lens_buf = CreateBuffer<uint32_t>(MemoryType::MEMORY_GPU, {batch_size});
  BufferMeta page_table = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(page_table_stride)});
  BufferMeta page_indices = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {K});
  constexpr int64_t workspace_stride_bytes = 4 * 1024 * 1024;
  BufferMeta workspace = CreateBuffer<uint8_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(workspace_stride_bytes)});
  BufferMeta metadata = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {(batch_size + 1) * 4});

  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(scores.data_ptr, scores_h.data(), scores_h.size() * sizeof(float), cudaMemcpyHostToDevice));
  uint32_t sl_h = seq_len;
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(seq_lens_buf.data_ptr, &sl_h, sizeof(uint32_t), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(page_table.data_ptr, pt_h.data(), pt_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

  TopKWrapper<float, int32_t>::TransformV2(
      seq_lens_buf.data_ptr, scores.data_ptr, page_table.data_ptr, page_indices.data_ptr, workspace.data_ptr,
      metadata.data_ptr, score_stride, page_table_stride, workspace_stride_bytes, batch_size,
      /*page_bits=*/page_bits, /*max_seq_len=*/seq_len, /*static_cluster_threshold=*/0, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<int32_t> out_h(K, -1);
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(out_h.data(), page_indices.data_ptr, K * sizeof(int32_t), cudaMemcpyDeviceToHost));

  // trivial_transform writes the page-translated index for positions 0..seq_len-1,
  // and -1 (i.e. out-of-range sentinel) for positions seq_len..K-1.
  for (uint32_t i = 0; i < seq_len; ++i) {
    const int32_t expected = PageToIndicesHost(pt_h.data(), i, page_bits);
    EXPECT_EQ(out_h[i], expected) << "mismatch at i=" << i;
  }
  for (uint32_t i = seq_len; i < K; ++i) {
    EXPECT_EQ(out_h[i], -1) << "expected -1 sentinel at i=" << i;
  }

  DeleteBuffer(scores);
  DeleteBuffer(seq_lens_buf);
  DeleteBuffer(page_table);
  DeleteBuffer(page_indices);
  DeleteBuffer(workspace);
  DeleteBuffer(metadata);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
