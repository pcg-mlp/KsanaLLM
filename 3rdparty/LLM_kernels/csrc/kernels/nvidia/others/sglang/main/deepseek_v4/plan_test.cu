/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Unit tests for the prefill-plan builder ported from SGLang DeepSeek-V4:
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/common.cuh
 */

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/plan.cuh"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {


// -----------------------------------------------------------------------------
// Golden reference implemented in plain C++ (mirrors `plan_prefill_cuda` exactly).
// -----------------------------------------------------------------------------
namespace {

struct HostPlanResult {
  std::vector<PrefillPlan> compress;
  std::vector<PrefillPlan> write;
};

HostPlanResult HostReferencePlan(const std::vector<IndicesT>& seq_lens, const std::vector<IndicesT>& extend_lens,
                                 uint32_t compress_ratio, bool is_overlap) {
  HostPlanResult r;
  const uint32_t ratio_eff = compress_ratio * (1u + (is_overlap ? 1u : 0u));
  uint32_t ragged_id = 0;
  const uint32_t batch_size = static_cast<uint32_t>(seq_lens.size());
  for (uint32_t b = 0; b < batch_size; ++b) {
    const uint32_t seq_len = static_cast<uint32_t>(seq_lens[b]);
    const uint32_t ext_len = static_cast<uint32_t>(extend_lens[b]);
    const uint32_t prefix_len = seq_len - ext_len;

    uint32_t start_write_pos = seq_len / compress_ratio * compress_ratio;
    if (is_overlap) {
      start_write_pos = (start_write_pos >= compress_ratio) ? (start_write_pos - compress_ratio) : 0u;
    }
    for (uint32_t j = 0; j < ext_len; ++j) {
      PrefillPlan p;
      p.ragged_id = ragged_id;
      p.batch_id = b;
      p.position = prefix_len + j;
      p.window_len = (j + 1u < ratio_eff) ? (ratio_eff - (j + 1u)) : 0u;
      if ((p.position + 1u) % compress_ratio == 0u) {
        r.compress.push_back(p);
      }
      if (p.position >= start_write_pos) {
        r.write.push_back(p);
      }
      ++ragged_id;
    }
  }
  return r;
}

// Sort + compare by (batch_id, position, ragged_id) since the GPU plan is
// produced via atomicAdd and ordering is not deterministic.
bool PlanEq(const PrefillPlan& a, const PrefillPlan& b) {
  return a.ragged_id == b.ragged_id && a.batch_id == b.batch_id && a.position == b.position &&
         a.window_len == b.window_len;
}

void SortPlans(std::vector<PrefillPlan>& v) {
  std::sort(v.begin(), v.end(), [](const PrefillPlan& a, const PrefillPlan& b) {
    if (a.batch_id != b.batch_id) return a.batch_id < b.batch_id;
    if (a.position != b.position) return a.position < b.position;
    return a.ragged_id < b.ragged_id;
  });
}

}  // namespace

class CompressorPlanTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  void RunOneCase(const std::vector<IndicesT>& seq_lens, const std::vector<IndicesT>& extend_lens,
                  uint32_t compress_ratio, bool is_overlap) {
    ASSERT_EQ(seq_lens.size(), extend_lens.size());
    const uint32_t batch_size = static_cast<uint32_t>(seq_lens.size());
    uint32_t num_tokens = 0;
    for (auto l : extend_lens) num_tokens += static_cast<uint32_t>(l);

    // ------ allocate device buffers ------
    BufferMeta seq_lens_d = CreateBuffer<IndicesT>(MemoryType::MEMORY_GPU, {batch_size});
    BufferMeta ext_lens_d = CreateBuffer<IndicesT>(MemoryType::MEMORY_GPU, {batch_size});
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(seq_lens_d.data_ptr, seq_lens.data(), batch_size * sizeof(IndicesT),
                                            cudaMemcpyHostToDevice, stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(ext_lens_d.data_ptr, extend_lens.data(), batch_size * sizeof(IndicesT),
                                            cudaMemcpyHostToDevice, stream));

    // Plan tensors shape: [num_tokens, kPrefillPlanDim] of uint8; equivalently
    // [num_tokens] of PrefillPlan (16 bytes).
    const size_t plan_bytes = static_cast<size_t>(num_tokens) * kPrefillPlanDim;
    BufferMeta compress_d = CreateBuffer<uint8_t>(MemoryType::MEMORY_GPU, {plan_bytes});
    BufferMeta write_d = CreateBuffer<uint8_t>(MemoryType::MEMORY_GPU, {plan_bytes});

    plan_prefill_compress(
        reinterpret_cast<PrefillPlan*>(compress_d.data_ptr), reinterpret_cast<PrefillPlan*>(write_d.data_ptr),
        reinterpret_cast<const IndicesT*>(seq_lens_d.data_ptr), reinterpret_cast<const IndicesT*>(ext_lens_d.data_ptr),
        batch_size, num_tokens, compress_ratio, is_overlap, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    HostPlanResult ref = HostReferencePlan(seq_lens, extend_lens, compress_ratio, is_overlap);

    // Read only the valid entries (compact plan, no padding).
    const size_t compress_count = ref.compress.size();
    const size_t write_count = ref.write.size();
    std::vector<PrefillPlan> got_compress(compress_count);
    std::vector<PrefillPlan> got_write(write_count);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(got_compress.data(), compress_d.data_ptr, compress_count * sizeof(PrefillPlan),
                                       cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(got_write.data(), write_d.data_ptr, write_count * sizeof(PrefillPlan), cudaMemcpyDeviceToHost));

    SortPlans(got_compress);
    SortPlans(got_write);
    SortPlans(ref.compress);
    SortPlans(ref.write);

    ASSERT_EQ(got_compress.size(), ref.compress.size());
    ASSERT_EQ(got_write.size(), ref.write.size());
    for (size_t i = 0; i < got_compress.size(); ++i) {
      EXPECT_TRUE(PlanEq(got_compress[i], ref.compress[i]))
          << "compress[" << i << "] mismatch: got {" << got_compress[i].ragged_id << "," << got_compress[i].batch_id
          << "," << got_compress[i].position << "," << got_compress[i].window_len << "}, ref {"
          << ref.compress[i].ragged_id << "," << ref.compress[i].batch_id << "," << ref.compress[i].position << ","
          << ref.compress[i].window_len << "}";
    }
    for (size_t i = 0; i < got_write.size(); ++i) {
      EXPECT_TRUE(PlanEq(got_write[i], ref.write[i]))
          << "write[" << i << "] mismatch: got {" << got_write[i].ragged_id << "," << got_write[i].batch_id << ","
          << got_write[i].position << "," << got_write[i].window_len << "}, ref {" << ref.write[i].ragged_id << ","
          << ref.write[i].batch_id << "," << ref.write[i].position << "," << ref.write[i].window_len << "}";
    }

    DeleteBuffer(seq_lens_d);
    DeleteBuffer(ext_lens_d);
    DeleteBuffer(compress_d);
    DeleteBuffer(write_d);
  }
};

// -----------------------------------------------------------------------------
// Sanity: PrefillPlan ABI must stay 16 bytes / 16-byte aligned.
// -----------------------------------------------------------------------------
TEST_F(CompressorPlanTestSuit, PrefillPlanLayoutIsStable) {
  EXPECT_EQ(sizeof(PrefillPlan), 16u);
  EXPECT_EQ(alignof(PrefillPlan), 16u);
  EXPECT_EQ(kPrefillPlanDim, 16);
}

// -----------------------------------------------------------------------------
// Case 1: pure-prefill (prefix_len == 0), is_overlap=true, ratio=4.
// -----------------------------------------------------------------------------
TEST_F(CompressorPlanTestSuit, PurePrefillOverlapRatio4) {
  std::vector<IndicesT> seq_lens = {8, 12, 5};
  std::vector<IndicesT> extend_lens = {8, 12, 5};
  RunOneCase(seq_lens, extend_lens, /*compress_ratio=*/4, /*is_overlap=*/true);
}

// -----------------------------------------------------------------------------
// Case 2: mixed prefix + extend tokens, is_overlap=false.
// -----------------------------------------------------------------------------
TEST_F(CompressorPlanTestSuit, MixedPrefixExtendNoOverlap) {
  std::vector<IndicesT> seq_lens = {16, 9, 24};
  std::vector<IndicesT> extend_lens = {8, 5, 12};
  RunOneCase(seq_lens, extend_lens, /*compress_ratio=*/4, /*is_overlap=*/false);
}

// -----------------------------------------------------------------------------
// Case 3: plan correctness without padding.
// -----------------------------------------------------------------------------
TEST_F(CompressorPlanTestSuit, NoPaddingPlanCorrectness) {
  std::vector<IndicesT> seq_lens = {7, 3};
  std::vector<IndicesT> extend_lens = {7, 3};
  RunOneCase(seq_lens, extend_lens, /*compress_ratio=*/4, /*is_overlap=*/true);
}

// -----------------------------------------------------------------------------
// Case 4: stress with random seq/extend lengths.
// -----------------------------------------------------------------------------
TEST_F(CompressorPlanTestSuit, RandomStress) {
  std::mt19937 rng(20260426);
  std::uniform_int_distribution<int> batch_dist(1, 16);
  std::uniform_int_distribution<int> ext_dist(1, 32);
  std::uniform_int_distribution<int> prefix_dist(0, 64);
  for (int trial = 0; trial < 8; ++trial) {
    const int batch = batch_dist(rng);
    std::vector<IndicesT> seq_lens(batch), extend_lens(batch);
    for (int i = 0; i < batch; ++i) {
      const int ext = ext_dist(rng);
      const int prefix = prefix_dist(rng);
      extend_lens[i] = ext;
      seq_lens[i] = static_cast<IndicesT>(prefix + ext);
    }
    const bool is_overlap = (trial & 1) != 0;
    RunOneCase(seq_lens, extend_lens, /*compress_ratio=*/4, is_overlap);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
