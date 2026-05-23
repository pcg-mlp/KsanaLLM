/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Unit tests for the DeepSeek-V4 Q+K RoPE kernel ported from SGLang:
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/rope.cuh
 */

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/rope.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {


// kRopeDim = 64 (fixed in the kernel)
static constexpr int64_t kRopeDim = 64;

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

// CPU reference for RoPE rotation on a single head (kRopeDim elements).
// freqs_cis: [max_pos, kRopeDim] bf16-rounded float, split layout:
//   first half is cos, second half is sin.
// inverse: if true, use conjugate (real*cos + imag*sin, -real*sin + imag*cos).
// is_neox: if true, split-half layout [0..31] = real, [32..63] = imag;
//          if false, interleaved layout [2p] = real, [2p+1] = imag.
void HostApplyRope(std::vector<float>& head, int64_t position, const std::vector<float>& freqs_cis, bool inverse,
                   bool is_neox) {
  const float* freq = freqs_cis.data() + position * kRopeDim;
  for (int64_t p = 0; p < kRopeDim / 2; ++p) {
    float x_re, x_im;
    if (is_neox) {
      x_re = head[p];
      x_im = head[kRopeDim / 2 + p];
    } else {
      x_re = head[2 * p];
      x_im = head[2 * p + 1];
    }
    const float f_cos = freq[p];                 // first half: cos
    const float f_sin = freq[kRopeDim / 2 + p];  // second half: sin
    float out_re, out_im;
    if (!inverse) {
      out_re = x_re * f_cos - x_im * f_sin;
      out_im = x_re * f_sin + x_im * f_cos;
    } else {
      out_re = x_re * f_cos + x_im * f_sin;
      out_im = -x_re * f_sin + x_im * f_cos;
    }
    if (is_neox) {
      head[p] = __bfloat162float(__float2bfloat16(out_re));
      head[kRopeDim / 2 + p] = __bfloat162float(__float2bfloat16(out_im));
    } else {
      head[2 * p] = __bfloat162float(__float2bfloat16(out_re));
      head[2 * p + 1] = __bfloat162float(__float2bfloat16(out_im));
    }
  }
}

}  // namespace

class CompressorRopeTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  void RunNumericalForwardTest(bool is_neox) {
    const uint32_t batch = 3;
    const uint32_t nq = 4;
    const uint32_t nk = 2;
    const int64_t maxpos = 16;
    const float eps = 5e-3f;

    std::mt19937 rng(20260426u);

    const size_t q_elem = static_cast<size_t>(batch) * nq * kRopeDim;
    const size_t k_elem = static_cast<size_t>(batch) * nk * kRopeDim;
    const size_t f_elem = static_cast<size_t>(maxpos) * kRopeDim;

    // Generate freqs_cis in split layout [cos | sin] directly as bf16
    std::vector<__nv_bfloat16> freqs_bf16_h(f_elem);
    std::vector<float> freqs_h(f_elem);
    {
      std::uniform_real_distribution<float> d(-1.0f, 1.0f);
      for (size_t i = 0; i < f_elem; ++i) {
        freqs_bf16_h[i] = __float2bfloat16(d(rng));
        freqs_h[i] = __bfloat162float(freqs_bf16_h[i]);
      }
    }

    // Generate Q and K as bf16-rounded floats
    std::vector<float> q_f(q_elem), k_f(k_elem);
    FillRandomBF16(&q_f, &rng);
    FillRandomBF16(&k_f, &rng);

    std::vector<__nv_bfloat16> q_h, k_h;
    HostFloatToBF16(q_f, &q_h);
    HostFloatToBF16(k_f, &k_h);
    HostBF16ToFloat(q_h, &q_f);
    HostBF16ToFloat(k_h, &k_f);

    // Positions (int64)
    std::vector<int64_t> pos_h(batch);
    for (uint32_t i = 0; i < batch; ++i) pos_h[i] = static_cast<int64_t>(i * 2);

    // Upload
    BufferMeta q_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {q_elem});
    BufferMeta k_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {k_elem});
    BufferMeta f_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {f_elem});
    BufferMeta pos_d = CreateBuffer<int64_t>(MemoryType::MEMORY_GPU, {batch});

    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(q_d.data_ptr, q_h.data(), q_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(k_d.data_ptr, k_h.data(), k_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(f_d.data_ptr, freqs_bf16_h.data(), f_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(pos_d.data_ptr, pos_h.data(), batch * sizeof(int64_t), cudaMemcpyHostToDevice));

    const int64_t q_stride_head = kRopeDim * static_cast<int64_t>(sizeof(__nv_bfloat16));
    const int64_t q_stride_batch = static_cast<int64_t>(nq) * q_stride_head;
    const int64_t k_stride_head = kRopeDim * static_cast<int64_t>(sizeof(__nv_bfloat16));
    const int64_t k_stride_batch = static_cast<int64_t>(nk) * k_stride_head;

    fused_qk_rope(q_d.data_ptr, k_d.data_ptr, f_d.data_ptr, pos_d.data_ptr, q_stride_batch, k_stride_batch,
                  q_stride_head, k_stride_head, nq, nk, batch, stream, /*inverse=*/false, /*is_neox=*/is_neox);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    std::vector<__nv_bfloat16> q_out_h(q_elem), k_out_h(k_elem);
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(q_out_h.data(), q_d.data_ptr, q_elem * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(k_out_h.data(), k_d.data_ptr, k_elem * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    std::vector<float> q_got, k_got;
    HostBF16ToFloat(q_out_h, &q_got);
    HostBF16ToFloat(k_out_h, &k_got);

    // CPU reference for Q
    std::vector<float> q_ref = q_f;
    for (uint32_t b = 0; b < batch; ++b) {
      for (uint32_t h = 0; h < nq; ++h) {
        const size_t off = (static_cast<size_t>(b) * nq + h) * kRopeDim;
        std::vector<float> head(q_ref.begin() + off, q_ref.begin() + off + kRopeDim);
        HostApplyRope(head, pos_h[b], freqs_h, false, is_neox);
        std::copy(head.begin(), head.end(), q_ref.begin() + off);
      }
    }

    // CPU reference for K
    std::vector<float> k_ref = k_f;
    for (uint32_t b = 0; b < batch; ++b) {
      for (uint32_t h = 0; h < nk; ++h) {
        const size_t off = (static_cast<size_t>(b) * nk + h) * kRopeDim;
        std::vector<float> head(k_ref.begin() + off, k_ref.begin() + off + kRopeDim);
        HostApplyRope(head, pos_h[b], freqs_h, false, is_neox);
        std::copy(head.begin(), head.end(), k_ref.begin() + off);
      }
    }

    // Check Q
    int q_mm = 0;
    for (size_t i = 0; i < q_elem; ++i) {
      if (std::fabs(q_got[i] - q_ref[i]) > eps) ++q_mm;
    }
    EXPECT_LE(q_mm, static_cast<int>(q_elem / 50)) << "Q mismatches=" << q_mm << " is_neox=" << is_neox;

    // Check K
    int k_mm = 0;
    for (size_t i = 0; i < k_elem; ++i) {
      if (std::fabs(k_got[i] - k_ref[i]) > eps) ++k_mm;
    }
    EXPECT_LE(k_mm, static_cast<int>(k_elem / 50)) << "K mismatches=" << k_mm << " is_neox=" << is_neox;

    DeleteBuffer(q_d);
    DeleteBuffer(k_d);
    DeleteBuffer(f_d);
    DeleteBuffer(pos_d);
  }

  void RunInverseUndoesForwardTest(bool is_neox) {
    const uint32_t batch = 2;
    const uint32_t nq = 2;
    const uint32_t nk = 1;
    const int64_t maxpos = 8;

    std::mt19937 rng(12345u);

    const size_t q_elem = static_cast<size_t>(batch) * nq * kRopeDim;
    const size_t f_elem = static_cast<size_t>(maxpos) * kRopeDim;

    // Generate freqs in split layout [cos | sin] directly as bf16
    // Use unit-complex values so that inverse exactly undoes forward.
    std::vector<__nv_bfloat16> freqs_bf16_h(f_elem);
    {
      std::uniform_real_distribution<float> angle_dist(-static_cast<float>(M_PI), static_cast<float>(M_PI));
      for (int64_t pos = 0; pos < maxpos; ++pos) {
        for (int64_t p = 0; p < kRopeDim / 2; ++p) {
          const float theta = angle_dist(rng);
          freqs_bf16_h[pos * kRopeDim + p] = __float2bfloat16(std::cos(theta));
          freqs_bf16_h[pos * kRopeDim + kRopeDim / 2 + p] = __float2bfloat16(std::sin(theta));
        }
      }
    }

    std::vector<float> q_f(q_elem);
    FillRandomBF16(&q_f, &rng);
    std::vector<float> q_orig = q_f;

    std::vector<__nv_bfloat16> q_h;
    HostFloatToBF16(q_f, &q_h);

    std::vector<int64_t> pos_h(batch);
    for (uint32_t i = 0; i < batch; ++i) pos_h[i] = static_cast<int64_t>(i);

    BufferMeta q_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {q_elem});
    BufferMeta f_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {f_elem});
    BufferMeta pos_d = CreateBuffer<int64_t>(MemoryType::MEMORY_GPU, {batch});

    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(q_d.data_ptr, q_h.data(), q_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(f_d.data_ptr, freqs_bf16_h.data(), f_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(pos_d.data_ptr, pos_h.data(), batch * sizeof(int64_t), cudaMemcpyHostToDevice));

    const int64_t q_stride_head = kRopeDim * static_cast<int64_t>(sizeof(__nv_bfloat16));
    const int64_t q_stride_batch = static_cast<int64_t>(nq) * q_stride_head;
    const int64_t k_stride_head = kRopeDim * static_cast<int64_t>(sizeof(__nv_bfloat16));
    const int64_t k_stride_batch = static_cast<int64_t>(nk) * k_stride_head;

    // Forward pass
    fused_qk_rope(q_d.data_ptr, nullptr, f_d.data_ptr, pos_d.data_ptr, q_stride_batch, k_stride_batch, q_stride_head,
                  k_stride_head, nq, 0u, batch, stream, /*inverse=*/false, /*is_neox=*/is_neox);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Inverse pass
    fused_qk_rope(q_d.data_ptr, nullptr, f_d.data_ptr, pos_d.data_ptr, q_stride_batch, k_stride_batch, q_stride_head,
                  k_stride_head, nq, 0u, batch, stream, /*inverse=*/true, /*is_neox=*/is_neox);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    std::vector<__nv_bfloat16> q_out_h(q_elem);
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(q_out_h.data(), q_d.data_ptr, q_elem * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    std::vector<float> q_got;
    HostBF16ToFloat(q_out_h, &q_got);

    // After forward + inverse, result should approximate the original (within bf16 rounding)
    int mismatches = 0;
    for (size_t i = 0; i < q_elem; ++i) {
      if (std::fabs(q_got[i] - q_orig[i]) > 2e-2f) ++mismatches;
    }
    EXPECT_LE(mismatches, static_cast<int>(q_elem / 20))
        << "Inverse did not undo forward: mismatches=" << mismatches << " is_neox=" << is_neox;

    DeleteBuffer(q_d);
    DeleteBuffer(f_d);
    DeleteBuffer(pos_d);
  }
};

// -----------------------------------------------------------------------------
// Test 1: smoke launch – no crash, no error
// -----------------------------------------------------------------------------
TEST_F(CompressorRopeTestSuit, SmokeLaunch) {
  const uint32_t batch = 4;
  const uint32_t nq = 8;
  const uint32_t nk = 2;
  const int64_t maxpos = 8;

  const size_t q_elem = static_cast<size_t>(batch) * nq * kRopeDim;
  const size_t k_elem = static_cast<size_t>(batch) * nk * kRopeDim;
  const size_t f_elem = static_cast<size_t>(maxpos) * kRopeDim;

  BufferMeta q_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {q_elem}, /*random*/ true);
  BufferMeta k_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {k_elem}, /*random*/ true);
  BufferMeta f_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {f_elem}, /*random*/ true);
  BufferMeta pos_d = CreateBuffer<int64_t>(MemoryType::MEMORY_GPU, {batch});

  std::vector<int64_t> pos_h(batch);
  for (uint32_t i = 0; i < batch; ++i) pos_h[i] = static_cast<int64_t>(i % maxpos);
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(pos_d.data_ptr, pos_h.data(), batch * sizeof(int64_t), cudaMemcpyHostToDevice));

  const int64_t q_stride_head = kRopeDim * static_cast<int64_t>(sizeof(__nv_bfloat16));
  const int64_t q_stride_batch = static_cast<int64_t>(nq) * q_stride_head;
  const int64_t k_stride_head = kRopeDim * static_cast<int64_t>(sizeof(__nv_bfloat16));
  const int64_t k_stride_batch = static_cast<int64_t>(nk) * k_stride_head;

  fused_qk_rope(q_d.data_ptr, k_d.data_ptr, f_d.data_ptr, pos_d.data_ptr, q_stride_batch, k_stride_batch, q_stride_head,
                k_stride_head, nq, nk, batch, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  DeleteBuffer(q_d);
  DeleteBuffer(k_d);
  DeleteBuffer(f_d);
  DeleteBuffer(pos_d);
}

// -----------------------------------------------------------------------------
// Test 2: empty batch is a no-op
// -----------------------------------------------------------------------------
TEST_F(CompressorRopeTestSuit, EmptyBatchIsNoOp) {
  fused_qk_rope(nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 4u, 1u, 0u, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// -----------------------------------------------------------------------------
// Test 3: numerical correctness – forward RoPE (interleaved)
// -----------------------------------------------------------------------------
TEST_F(CompressorRopeTestSuit, NumericalForward) { RunNumericalForwardTest(/*is_neox=*/false); }

// -----------------------------------------------------------------------------
// Test 4: inverse RoPE undoes forward RoPE (interleaved)
// -----------------------------------------------------------------------------
TEST_F(CompressorRopeTestSuit, InverseUndoesForward) { RunInverseUndoesForwardTest(/*is_neox=*/false); }

// -----------------------------------------------------------------------------
// Test 5: NeoX style – numerical correctness (split-half)
// -----------------------------------------------------------------------------
TEST_F(CompressorRopeTestSuit, NumericalForwardNeox) { RunNumericalForwardTest(/*is_neox=*/true); }

// -----------------------------------------------------------------------------
// Test 6: NeoX style – inverse undoes forward (split-half)
// -----------------------------------------------------------------------------
TEST_F(CompressorRopeTestSuit, InverseUndoesForwardNeox) { RunInverseUndoesForwardTest(/*is_neox=*/true); }

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
