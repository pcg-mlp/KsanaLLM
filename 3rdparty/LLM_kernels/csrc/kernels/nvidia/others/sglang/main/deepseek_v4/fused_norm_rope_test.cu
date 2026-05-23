/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Unit tests for the fused RMSNorm+RoPE kernel ported from SGLang DeepSeek-V4:
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope.cuh
 */

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/fused_norm_rope.cuh"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/plan.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {


static constexpr int64_t kFNRRopeDim = 64;

// -----------------------------------------------------------------------------
// CPU reference helpers
// -----------------------------------------------------------------------------
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

void HostBF16ToFloat(const std::vector<__nv_bfloat16>& src, std::vector<float>* dst) {
  dst->resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) (*dst)[i] = __bfloat162float(src[i]);
}

// Apply RMSNorm in-place on a single head of `head_dim` elements with the given weight.
void HostRmsNormInplace(std::vector<float>& head, const std::vector<float>& weight, int64_t head_dim, float eps) {
  float ss = 0.0f;
  for (int64_t i = 0; i < head_dim; ++i) ss += head[i] * head[i];
  const float factor = 1.0f / std::sqrt(ss / static_cast<float>(head_dim) + eps);
  for (int64_t i = 0; i < head_dim; ++i) {
    head[i] = __bfloat162float(__float2bfloat16(head[i] * factor * weight[i]));
  }
}

// Apply RoPE in-place on the last kRopeDim elements of a head.
// freqs_cis layout (split): [cos_0..cos_{N/2-1}, sin_0..sin_{N/2-1}] per position, stored as bf16 floats.
void HostApplyRopeBF16(std::vector<float>& head, int64_t head_dim, int32_t position,
                       const std::vector<float>& freqs_cis) {
  const float* freq_base = freqs_cis.data() + position * kFNRRopeDim;
  const float* freq_cos = freq_base;                    // first half: cos values
  const float* freq_sin = freq_base + kFNRRopeDim / 2;  // second half: sin values
  const int64_t rope_start = head_dim - kFNRRopeDim;
  for (int64_t p = 0; p < kFNRRopeDim / 2; ++p) {
    const float x_re = head[rope_start + 2 * p];
    const float x_im = head[rope_start + 2 * p + 1];
    const float f_re = freq_cos[p];
    const float f_im = freq_sin[p];
    head[rope_start + 2 * p] = __bfloat162float(__float2bfloat16(x_re * f_re - x_im * f_im));
    head[rope_start + 2 * p + 1] = __bfloat162float(__float2bfloat16(x_re * f_im + x_im * f_re));
  }
}

// Full CPU reference for fused_norm_rope
void HostFusedNormRope(std::vector<float>& input /*[num_works * head_dim]*/,
                       const std::vector<float>& weight /*[head_dim]*/,
                       const std::vector<float>& freqs_cis /*[max_pos, rope_dim]*/, int64_t head_dim, float eps,
                       const std::vector<int32_t>& positions /*[num_works]*/) {
  const uint32_t nw = static_cast<uint32_t>(positions.size());
  for (uint32_t w = 0; w < nw; ++w) {
    std::vector<float> head(input.begin() + w * head_dim, input.begin() + (w + 1) * head_dim);
    HostRmsNormInplace(head, weight, head_dim, eps);
    HostApplyRopeBF16(head, head_dim, positions[w], freqs_cis);
    std::copy(head.begin(), head.end(), input.begin() + w * head_dim);
  }
}

}  // namespace

class CompressorFusedNormRopeTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
};

// -----------------------------------------------------------------------------
// Test 1: smoke launch for all modes and dtype/dim combinations
// -----------------------------------------------------------------------------
TEST_F(CompressorFusedNormRopeTestSuit, SmokeLaunchAllModes) {
  constexpr int64_t head_dim = 128;
  const uint32_t num_works = 4;
  const int64_t max_pos = 16;

  const size_t in_elem = static_cast<size_t>(num_works) * head_dim;
  const size_t f_elem = static_cast<size_t>(max_pos) * kFNRRopeDim;

  BufferMeta in_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {in_elem}, /*random*/ true);
  BufferMeta w_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)head_dim}, /*random*/ true);
  BufferMeta f_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {f_elem}, /*random*/ true);

  // DefaultForward: handle is array of int64_t positions
  BufferMeta pos_d = CreateBuffer<int64_t>(MemoryType::MEMORY_GPU, {num_works});
  std::vector<int64_t> pos_h(num_works, 1LL);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(pos_d.data_ptr, pos_h.data(), num_works * sizeof(int64_t), cudaMemcpyHostToDevice));

  compress_fused_norm_rope_inplace<__nv_bfloat16, head_dim>(in_d.data_ptr, w_d.data_ptr, pos_d.data_ptr, f_d.data_ptr,
                                                            num_works, ForwardMode::DefaultForward, 1e-5f,
                                                            /*compress_ratio*/ 128u, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  // CompressDecode: handle is array of int32_t seq_lens
  BufferMeta sl_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {num_works});
  // seq_lens must be multiples of compress_ratio=128 for the kernel to run; otherwise skip.
  std::vector<int32_t> sl_h = {128, 256, 128, 256};
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(sl_d.data_ptr, sl_h.data(), num_works * sizeof(int32_t), cudaMemcpyHostToDevice));

  compress_fused_norm_rope_inplace<__nv_bfloat16, head_dim>(in_d.data_ptr, w_d.data_ptr, sl_d.data_ptr, f_d.data_ptr,
                                                            num_works, ForwardMode::CompressDecode, 1e-5f, 128u,
                                                            stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  // CompressExtend: handle is array of PrefillPlan
  BufferMeta plan_d = CreateBuffer<uint8_t>(MemoryType::MEMORY_GPU, {num_works * sizeof(PrefillPlan)});
  std::vector<PrefillPlan> plans(num_works);
  for (uint32_t i = 0; i < num_works; ++i) {
    plans[i].ragged_id = i;
    plans[i].batch_id = 0;
    plans[i].position = 128 * (i + 1);  // position >= compress_ratio
    plans[i].window_len = 128;
  }
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(plan_d.data_ptr, plans.data(), num_works * sizeof(PrefillPlan), cudaMemcpyHostToDevice));

  compress_fused_norm_rope_inplace<__nv_bfloat16, head_dim>(in_d.data_ptr, w_d.data_ptr, plan_d.data_ptr, f_d.data_ptr,
                                                            num_works, ForwardMode::CompressExtend, 1e-5f, 128u,
                                                            stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  DeleteBuffer(in_d);
  DeleteBuffer(w_d);
  DeleteBuffer(f_d);
  DeleteBuffer(pos_d);
  DeleteBuffer(sl_d);
  DeleteBuffer(plan_d);
}

// -----------------------------------------------------------------------------
// Test 2: empty num_works is a no-op
// -----------------------------------------------------------------------------
TEST_F(CompressorFusedNormRopeTestSuit, EmptyWorksIsNoOp) {
  compress_fused_norm_rope_inplace<__nv_bfloat16, 128>(nullptr, nullptr, nullptr, nullptr, 0u,
                                                       ForwardMode::DefaultForward, 1e-5f, 128u, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// -----------------------------------------------------------------------------
// Test 3: DefaultForward mode numerical correctness
// -----------------------------------------------------------------------------
TEST_F(CompressorFusedNormRopeTestSuit, DefaultForwardNumerical) {
  constexpr int64_t head_dim = 128;
  const uint32_t num_works = 4;
  const int64_t max_pos = 16;
  const float eps = 1e-5f;

  std::mt19937 rng(20260426u);

  const size_t in_elem = static_cast<size_t>(num_works) * head_dim;
  const size_t w_elem = static_cast<size_t>(head_dim);
  const size_t f_elem = static_cast<size_t>(max_pos) * kFNRRopeDim;

  std::vector<float> in_f(in_elem), w_f(w_elem), freq_f(f_elem);
  FillRandomBF16(&in_f, &rng);
  FillRandomBF16(&w_f, &rng);
  FillRandomBF16(&freq_f, &rng);

  std::vector<__nv_bfloat16> in_h, w_h, freq_h;
  HostFloatToBF16(in_f, &in_h);
  HostFloatToBF16(w_f, &w_h);
  HostFloatToBF16(freq_f, &freq_h);
  HostBF16ToFloat(in_h, &in_f);
  HostBF16ToFloat(w_h, &w_f);
  HostBF16ToFloat(freq_h, &freq_f);

  std::vector<int64_t> pos_h(num_works);
  for (uint32_t i = 0; i < num_works; ++i) pos_h[i] = static_cast<int64_t>(i + 1);

  BufferMeta in_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {in_elem});
  BufferMeta w_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {w_elem});
  BufferMeta f_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {f_elem});
  BufferMeta pos_d = CreateBuffer<int64_t>(MemoryType::MEMORY_GPU, {num_works});

  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(in_d.data_ptr, in_h.data(), in_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(w_d.data_ptr, w_h.data(), w_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(f_d.data_ptr, freq_h.data(), f_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(pos_d.data_ptr, pos_h.data(), num_works * sizeof(int64_t), cudaMemcpyHostToDevice));

  compress_fused_norm_rope_inplace<__nv_bfloat16, head_dim>(in_d.data_ptr, w_d.data_ptr, pos_d.data_ptr, f_d.data_ptr,
                                                            num_works, ForwardMode::DefaultForward, eps, 128u, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> out_h(in_elem);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(out_h.data(), in_d.data_ptr, in_elem * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  std::vector<float> got;
  HostBF16ToFloat(out_h, &got);

  // CPU reference
  std::vector<int32_t> pos32(num_works);
  for (uint32_t i = 0; i < num_works; ++i) pos32[i] = static_cast<int32_t>(pos_h[i]);
  HostFusedNormRope(in_f, w_f, freq_f, head_dim, eps, pos32);

  int mismatches = 0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < in_elem; ++i) {
    const float diff = std::fabs(got[i] - in_f[i]);
    max_abs = std::max(max_abs, diff);
    if (diff > 1e-2f) ++mismatches;
  }
  EXPECT_LE(mismatches, static_cast<int>(in_elem / 30))
      << "DefaultForward: max_abs=" << max_abs << " mismatches=" << mismatches;

  DeleteBuffer(in_d);
  DeleteBuffer(w_d);
  DeleteBuffer(f_d);
  DeleteBuffer(pos_d);
}

// -----------------------------------------------------------------------------
// Test 4: CompressDecode mode – items with seq_len not divisible by compress_ratio
//         must remain unchanged in output.
// -----------------------------------------------------------------------------
TEST_F(CompressorFusedNormRopeTestSuit, CompressDecodeSkipsNonAligned) {
  constexpr int64_t head_dim = 128;
  const uint32_t num_works = 4;
  const uint32_t compress_ratio = 128u;

  std::mt19937 rng(99u);
  const size_t in_elem = static_cast<size_t>(num_works) * head_dim;
  std::vector<float> in_f(in_elem);
  FillRandomBF16(&in_f, &rng);
  std::vector<float> in_orig = in_f;

  std::vector<__nv_bfloat16> in_h;
  HostFloatToBF16(in_f, &in_h);

  // Mix of aligned (128, 256) and non-aligned (1, 127) seq_lens
  std::vector<int32_t> sl_h = {128, 1, 256, 127};

  BufferMeta in_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {in_elem});
  BufferMeta w_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)head_dim}, /*random*/ true);
  BufferMeta f_d =
      CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {static_cast<size_t>(512) * kFNRRopeDim}, /*random*/ true);
  BufferMeta sl_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {num_works});

  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(in_d.data_ptr, in_h.data(), in_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(sl_d.data_ptr, sl_h.data(), num_works * sizeof(int32_t), cudaMemcpyHostToDevice));

  compress_fused_norm_rope_inplace<__nv_bfloat16, head_dim>(in_d.data_ptr, w_d.data_ptr, sl_d.data_ptr, f_d.data_ptr,
                                                            num_works, ForwardMode::CompressDecode, 1e-5f,
                                                            compress_ratio, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> out_h(in_elem);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(out_h.data(), in_d.data_ptr, in_elem * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  std::vector<float> got;
  HostBF16ToFloat(out_h, &got);
  HostBF16ToFloat(in_h, &in_f);  // recover exact bf16 values

  // Non-aligned indices: work_id=1 (seq_len=1) and work_id=3 (seq_len=127) must be unchanged.
  for (int w : {1, 3}) {
    for (int64_t i = 0; i < head_dim; ++i) {
      const size_t idx = static_cast<size_t>(w) * head_dim + i;
      EXPECT_FLOAT_EQ(got[idx], in_f[idx]) << "work=" << w << " i=" << i << " should be unchanged";
    }
  }

  DeleteBuffer(in_d);
  DeleteBuffer(w_d);
  DeleteBuffer(f_d);
  DeleteBuffer(sl_d);
}

// -----------------------------------------------------------------------------
// Test 5: CompressExtend mode – sentinel ragged_id == 0xFFFFFFFF must be skipped
// -----------------------------------------------------------------------------
TEST_F(CompressorFusedNormRopeTestSuit, CompressExtendSentinelSkipped) {
  constexpr int64_t head_dim = 128;
  const uint32_t num_works = 3;
  const uint32_t compress_ratio = 128u;

  std::mt19937 rng(54321u);
  const size_t in_elem = static_cast<size_t>(num_works) * head_dim;
  std::vector<float> in_f(in_elem);
  FillRandomBF16(&in_f, &rng);
  std::vector<float> in_orig = in_f;
  std::vector<__nv_bfloat16> in_h;
  HostFloatToBF16(in_f, &in_h);

  // plan[1] is a sentinel
  std::vector<PrefillPlan> plans(num_works);
  plans[0].ragged_id = 0;
  plans[0].batch_id = 0;
  plans[0].position = 128;
  plans[0].window_len = 128;
  plans[1].ragged_id = 0xFFFFFFFFu;  // sentinel → skip
  plans[2].ragged_id = 2;
  plans[2].batch_id = 0;
  plans[2].position = 256;
  plans[2].window_len = 128;

  BufferMeta in_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {in_elem});
  BufferMeta w_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)head_dim}, /*random*/ true);
  BufferMeta f_d =
      CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {static_cast<size_t>(512) * kFNRRopeDim}, /*random*/ true);
  BufferMeta plan_d = CreateBuffer<uint8_t>(MemoryType::MEMORY_GPU, {num_works * sizeof(PrefillPlan)});

  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(in_d.data_ptr, in_h.data(), in_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(plan_d.data_ptr, plans.data(), num_works * sizeof(PrefillPlan), cudaMemcpyHostToDevice));

  compress_fused_norm_rope_inplace<__nv_bfloat16, head_dim>(in_d.data_ptr, w_d.data_ptr, plan_d.data_ptr, f_d.data_ptr,
                                                            num_works, ForwardMode::CompressExtend, 1e-5f,
                                                            compress_ratio, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> out_h(in_elem);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(out_h.data(), in_d.data_ptr, in_elem * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  std::vector<float> got;
  HostBF16ToFloat(out_h, &got);
  HostBF16ToFloat(in_h, &in_f);

  // work_id=1 maps to sentinel plan → input row 1 (ragged_id=1) not written
  // The sentinel work uses input pointer for ragged_id=0xFFFFFFFF which is way out of bounds,
  // but the kernel must early-return before accessing memory.
  // We can only verify the kernel didn't crash; the output for row 1 may be arbitrary.
  // (We check that rows 0 and 2 were modified, i.e. the kernel did run for them.)
  bool row0_changed = false, row2_changed = false;
  for (int64_t i = 0; i < head_dim; ++i) {
    if (got[i] != in_f[i]) row0_changed = true;
    if (got[2 * head_dim + i] != in_f[2 * head_dim + i]) row2_changed = true;
  }
  EXPECT_TRUE(row0_changed) << "Row 0 should have been modified by CompressExtend";
  EXPECT_TRUE(row2_changed) << "Row 2 should have been modified by CompressExtend";

  DeleteBuffer(in_d);
  DeleteBuffer(w_d);
  DeleteBuffer(f_d);
  DeleteBuffer(plan_d);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
