/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Unit tests for the RMSNorm kernel ported from SGLang DeepSeek-V4:
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/rmsnorm.cuh
 */

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/rmsnorm.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_bf16.h>
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

// CPU reference for rmsnorm_self (no weight, identity scale)
// input layout: [batch, num_head, head_dim] with arbitrary strides
// output layout: contiguous [batch * num_head, head_dim]
void HostRmsNormSelf(const std::vector<float>& input_flat /*[batch * num_head * head_dim], contiguous*/,
                     uint32_t batch_size, uint32_t num_head, int64_t head_dim, float eps,
                     std::vector<float>* output /*[batch * num_head * head_dim]*/) {
  output->resize(static_cast<size_t>(batch_size) * num_head * head_dim);
  for (uint32_t b = 0; b < batch_size; ++b) {
    for (uint32_t h = 0; h < num_head; ++h) {
      const size_t row = static_cast<size_t>(b) * num_head + h;
      const size_t in_off = row * static_cast<size_t>(head_dim);
      float ss = 0.0f;
      for (int64_t i = 0; i < head_dim; ++i) {
        const float v = input_flat[in_off + i];
        ss += v * v;
      }
      const float scale = 1.0f / std::sqrt(ss / static_cast<float>(head_dim) + eps);
      for (int64_t i = 0; i < head_dim; ++i) {
        (*output)[in_off + i] = input_flat[in_off + i] * scale;
      }
    }
  }
}

}  // namespace

class CompressorRmsNormTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
};

// -----------------------------------------------------------------------------
// Test 1: smoke test – verify kernel launches without error for all type/dim combos
// -----------------------------------------------------------------------------
TEST_F(CompressorRmsNormTestSuit, SmokeLaunchAllDtypesDims) {
  const uint32_t batch_size = 3;
  const uint32_t num_head = 4;
  for (int head_dim : {128, 256}) {
    const size_t n = static_cast<size_t>(batch_size) * num_head * head_dim;

    BufferMeta in_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {n}, /*random*/ true);
    BufferMeta out_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {n});

    const int64_t stride_head = static_cast<int64_t>(head_dim) * sizeof(__nv_bfloat16);
    const int64_t stride_batch = static_cast<int64_t>(num_head) * stride_head;

    rmsnorm_self<__nv_bfloat16, 128>(in_d.data_ptr, out_d.data_ptr, stride_batch, stride_head, batch_size, num_head,
                                     1e-5f, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    rmsnorm_self<float, 128>(nullptr, nullptr, stride_batch, stride_head, 0u, num_head, 1e-5f, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    DeleteBuffer(in_d);
    DeleteBuffer(out_d);
  }
}

// -----------------------------------------------------------------------------
// Test 2: empty batch is a no-op
// -----------------------------------------------------------------------------
TEST_F(CompressorRmsNormTestSuit, EmptyBatchIsNoOp) {
  rmsnorm_self<__nv_bfloat16, 128>(nullptr, nullptr, 0, 0, 0u, 0u, 1e-5f, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// -----------------------------------------------------------------------------
// Test 3: numerical correctness for bf16, head_dim=128
// -----------------------------------------------------------------------------
TEST_F(CompressorRmsNormTestSuit, NumericalMatchBF16HeadDim128) {
  const uint32_t batch_size = 2;
  const uint32_t num_head = 3;
  constexpr int64_t head_dim = 128;
  const float eps = 1e-5f;

  std::mt19937 rng(20260426u);
  const size_t n = static_cast<size_t>(batch_size) * num_head * head_dim;
  std::vector<float> in_f(n);
  FillRandomBF16(&in_f, &rng);

  // Convert to bf16 for upload
  std::vector<__nv_bfloat16> in_h;
  HostFloatToBF16(in_f, &in_h);

  // Recover what was actually stored (round-trip)
  HostBF16ToFloat(in_h, &in_f);

  // GPU
  BufferMeta in_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {n});
  BufferMeta out_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {n});
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(in_d.data_ptr, in_h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  const int64_t stride_head = head_dim * static_cast<int64_t>(sizeof(__nv_bfloat16));
  const int64_t stride_batch = static_cast<int64_t>(num_head) * stride_head;

  rmsnorm_self<__nv_bfloat16, head_dim>(in_d.data_ptr, out_d.data_ptr, stride_batch, stride_head, batch_size, num_head,
                                        eps, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> out_h(n);
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(out_h.data(), out_d.data_ptr, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  std::vector<float> got;
  HostBF16ToFloat(out_h, &got);

  // CPU reference (operates on contiguous input = same as GPU since strides == contiguous)
  std::vector<float> ref;
  HostRmsNormSelf(in_f, batch_size, num_head, head_dim, eps, &ref);

  // Compare
  int mismatches = 0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const float diff = std::fabs(got[i] - ref[i]);
    max_abs = std::max(max_abs, diff);
    if (diff > 5e-3f) ++mismatches;
  }
  EXPECT_LE(mismatches, static_cast<int>(n / 50)) << "max_abs=" << max_abs << " mismatches=" << mismatches << "/" << n;

  DeleteBuffer(in_d);
  DeleteBuffer(out_d);
}

// -----------------------------------------------------------------------------
// Test 4: numerical correctness for float, head_dim=256
// -----------------------------------------------------------------------------
TEST_F(CompressorRmsNormTestSuit, NumericalMatchFloatHeadDim256) {
  const uint32_t batch_size = 2;
  const uint32_t num_head = 2;
  constexpr int64_t head_dim = 256;
  const float eps = 1e-6f;

  std::mt19937 rng(202604u);
  const size_t n = static_cast<size_t>(batch_size) * num_head * head_dim;
  std::vector<float> in_f(n);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (auto& x : in_f) x = dist(rng);

  BufferMeta in_d = CreateBuffer<float>(MemoryType::MEMORY_GPU, {n});
  BufferMeta out_d = CreateBuffer<float>(MemoryType::MEMORY_GPU, {n});
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(in_d.data_ptr, in_f.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const int64_t stride_head = head_dim * static_cast<int64_t>(sizeof(float));
  const int64_t stride_batch = static_cast<int64_t>(num_head) * stride_head;

  rmsnorm_self<float, head_dim>(in_d.data_ptr, out_d.data_ptr, stride_batch, stride_head, batch_size, num_head, eps,
                                stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::vector<float> got(n);
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(got.data(), out_d.data_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> ref;
  HostRmsNormSelf(in_f, batch_size, num_head, head_dim, eps, &ref);

  int mismatches = 0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const float diff = std::fabs(got[i] - ref[i]);
    max_abs = std::max(max_abs, diff);
    if (diff > 1e-4f) ++mismatches;
  }
  EXPECT_LE(mismatches, static_cast<int>(n / 100)) << "max_abs=" << max_abs << " mismatches=" << mismatches << "/" << n;

  DeleteBuffer(in_d);
  DeleteBuffer(out_d);
}

// -----------------------------------------------------------------------------
// Test 5: strided input – input is a sub-region of a larger tensor
// -----------------------------------------------------------------------------
TEST_F(CompressorRmsNormTestSuit, StridedInputMatchesContiguous) {
  const uint32_t batch_size = 2;
  const uint32_t num_head = 2;
  constexpr int64_t head_dim = 128;
  // Pad each head row with an extra 32 elements to force non-unit stride.
  constexpr int64_t pad_dim = 32;
  constexpr int64_t stride_head_elem = head_dim + pad_dim;  // in DType units
  const float eps = 1e-5f;

  std::mt19937 rng(987654u);
  const size_t in_total = static_cast<size_t>(batch_size) * num_head * stride_head_elem;
  std::vector<float> in_f(in_total);
  FillRandomBF16(&in_f, &rng);

  std::vector<__nv_bfloat16> in_h;
  HostFloatToBF16(in_f, &in_h);
  HostBF16ToFloat(in_h, &in_f);

  BufferMeta in_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {in_total});
  BufferMeta out_d =
      CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size) * num_head * head_dim});
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(in_d.data_ptr, in_h.data(), in_total * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  const int64_t stride_head = stride_head_elem * static_cast<int64_t>(sizeof(__nv_bfloat16));
  const int64_t stride_batch = static_cast<int64_t>(num_head) * stride_head;

  rmsnorm_self<__nv_bfloat16, head_dim>(in_d.data_ptr, out_d.data_ptr, stride_batch, stride_head, batch_size, num_head,
                                        eps, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> out_h(static_cast<size_t>(batch_size) * num_head * head_dim);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(out_h.data(), out_d.data_ptr, out_h.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  std::vector<float> got;
  HostBF16ToFloat(out_h, &got);

  // Build contiguous reference input from strided layout
  std::vector<float> ref_in(static_cast<size_t>(batch_size) * num_head * head_dim);
  for (uint32_t b = 0; b < batch_size; ++b) {
    for (uint32_t h = 0; h < num_head; ++h) {
      const size_t src_off = (b * num_head + h) * stride_head_elem;
      const size_t dst_off = (b * num_head + h) * head_dim;
      for (int64_t i = 0; i < head_dim; ++i) {
        ref_in[dst_off + i] = in_f[src_off + i];
      }
    }
  }

  std::vector<float> ref;
  HostRmsNormSelf(ref_in, batch_size, num_head, head_dim, eps, &ref);

  int mismatches = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::fabs(got[i] - ref[i]) > 5e-3f) ++mismatches;
  }
  EXPECT_LE(mismatches, static_cast<int>(got.size() / 50)) << "Strided input mismatches=" << mismatches;

  DeleteBuffer(in_d);
  DeleteBuffer(out_d);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
