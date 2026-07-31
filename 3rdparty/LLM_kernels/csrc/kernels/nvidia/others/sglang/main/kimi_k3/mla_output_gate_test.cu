/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Unit tests for the Kimi-K3 MLA output gate kernel.
 */

#include "csrc/kernels/nvidia/others/sglang/main/kimi_k3/mla_output_gate.h"

#include <cmath>
#include <cstdint>
#include <cstring>
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

// -----------------------------------------------------------------------------
// CPU reference helpers
// -----------------------------------------------------------------------------
namespace {

// Relative size of one representable step (bf16 keeps 8 explicit mantissa bits).
constexpr float kBF16Step = 1.0f / 128.0f;

// out = bf16(fp32(x) * fp32(bf16(sigmoid(fp32(g))))).
// Reproduces the double rounding of the unfused `torch.sigmoid(gate) * x` pair,
// which is exactly what the kernel is required to match.
__nv_bfloat16 HostGateOne(__nv_bfloat16 x, __nv_bfloat16 g) {
  const float gf = __bfloat162float(g);
  const __nv_bfloat16 s = __float2bfloat16(1.0f / (1.0f + std::exp(-gf)));
  return __float2bfloat16(__bfloat162float(x) * __bfloat162float(s));
}

bool SameBits(__nv_bfloat16 a, __nv_bfloat16 b) {
  uint16_t ba, bb;
  std::memcpy(&ba, &a, sizeof(uint16_t));
  std::memcpy(&bb, &b, sizeof(uint16_t));
  return ba == bb;
}

}  // namespace

class MlaOutputGateTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  // Runs the kernel over `num_elements` and compares against the CPU reference.
  //   in_place - write the result back over `x` instead of a separate buffer.
  void RunNumericalTest(size_t num_elements, bool in_place = false) {
    std::mt19937 rng(20260730u + static_cast<uint32_t>(num_elements));
    // Wide enough to cover the saturating tails of sigmoid.
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);

    std::vector<__nv_bfloat16> x_h(num_elements), g_h(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
      x_h[i] = __float2bfloat16(dist(rng));
      g_h[i] = __float2bfloat16(dist(rng));
    }

    BufferMeta x_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {num_elements});
    BufferMeta g_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {num_elements});
    BufferMeta o_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {num_elements});

    void* out_p = in_place ? x_d.data_ptr : o_d.data_ptr;
    const size_t bytes = num_elements * sizeof(__nv_bfloat16);

    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(x_d.data_ptr, x_h.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(g_d.data_ptr, g_h.data(), bytes, cudaMemcpyHostToDevice));

    mla_output_gate(x_d.data_ptr, g_d.data_ptr, out_p, num_elements, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<__nv_bfloat16> got(num_elements);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(got.data(), out_p, bytes, cudaMemcpyDeviceToHost));

    size_t exact = 0;
    for (size_t i = 0; i < num_elements; ++i) {
      const __nv_bfloat16 ref = HostGateOne(x_h[i], g_h[i]);
      if (SameBits(got[i], ref)) {
        ++exact;
        continue;
      }
      // A miss is only tolerated when the fp32 sigmoid landed on a rounding tie:
      // the value must still be within one representable step of the reference.
      const float got_f = __bfloat162float(got[i]);
      const float ref_f = __bfloat162float(ref);
      EXPECT_LE(std::fabs(got_f - ref_f), std::fabs(ref_f) * kBF16Step + 1e-6f)
          << "element " << i << ": got=" << got_f << " ref=" << ref_f;
    }
    // The double-rounding contract means bit parity, not just closeness -- hold
    // the kernel to that. The tolerance branch above only exists so a failure
    // reports how far off it drifted instead of just "not equal".
    EXPECT_GE(exact, num_elements) << "bit-exact " << exact << "/" << num_elements << " (n=" << num_elements << ")";

    DeleteBuffer(x_d);
    DeleteBuffer(g_d);
    DeleteBuffer(o_d);
  }
};

// -----------------------------------------------------------------------------
// Test 1: zero elements is a no-op, even with null pointers
// -----------------------------------------------------------------------------
TEST_F(MlaOutputGateTestSuit, EmptyIsNoOp) {
  mla_output_gate(nullptr, nullptr, nullptr, 0, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// -----------------------------------------------------------------------------
// Test 2: numerical correctness, exactly one block worth of vectors
// -----------------------------------------------------------------------------
TEST_F(MlaOutputGateTestSuit, NumericalSingleBlock) { RunNumericalTest(/*num_elements=*/256 * 8); }

// -----------------------------------------------------------------------------
// Test 3: numerical correctness across many blocks, last block partly idle
// -----------------------------------------------------------------------------
TEST_F(MlaOutputGateTestSuit, NumericalMultiBlockRagged) { RunNumericalTest(/*num_elements=*/8200); }

// -----------------------------------------------------------------------------
// Test 4: out aliases x -- the in-place form used by the MLA forward pass
// -----------------------------------------------------------------------------
TEST_F(MlaOutputGateTestSuit, NumericalInPlace) { RunNumericalTest(/*num_elements=*/4096, /*in_place=*/true); }

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
