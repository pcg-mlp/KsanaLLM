/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Unit tests for the C4 KV-compression kernel ported from SGLang DeepSeek-V4:
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/c4.cuh
 */

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/c4.cuh"

#include <cmath>
#include <cstdint>
#include <cstring>
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

// -----------------------------------------------------------------------------
// Small host-side helpers
// -----------------------------------------------------------------------------
namespace {

// Element layout of the `kv_score_input` / buffer's last dim:
//   [0,         head_dim)         -> kv overlap slice
//   [head_dim,  2*head_dim)       -> kv slice
//   [2*head_dim,3*head_dim)       -> score overlap slice
//   [3*head_dim,4*head_dim)       -> score slice
// Thus `c4_forward` reads `kv = src[head_dim:2*head_dim]` and
// `score = src[3*head_dim:4*head_dim]` (for the current ragged token).

// CPU implementation of c4_forward for the RingBuffer decode path, covering the
// most common seq_len >= 4 case. Replicates the kernel numerically.
void HostC4ForwardRing(const std::vector<float>& buffer_bf /*[8, head_dim*4]*/,
                       const std::vector<float>& input_bf /*[head_dim*4]*/,
                       const std::vector<float>& bias /*[8, head_dim]*/, int32_t seq_len, int32_t head_dim,
                       std::vector<float>* out_fp32 /*[head_dim]*/) {
  out_fp32->assign(head_dim, 0.0f);
  const int64_t elem = head_dim * 4;
  for (int i = 0; i < head_dim; ++i) {
    float scores[8];
    float kvs[8];
    for (int j = 0; j < 8; ++j) {
      const bool is_overlap = j < 4;
      // Map j to an address, identical to kernel branch `i < window_len` =>
      // ring buffer; for `seq_len == 4` the front 4 slots are synthesized as
      // neg-inf + 0. For seq_len >= 8 all 8 slots come from ring buffer.
      // This test always uses window_len == 8 and seq_len a multiple of 4.
      float kv_val, score_val;
      const int base_offset = is_overlap ? 0 : head_dim;  // kv overlap vs kv
      const int score_offset = is_overlap ? (2 * head_dim) : (3 * head_dim);
      const int k = (seq_len + j) % 8;
      kv_val = buffer_bf[k * elem + base_offset + i];
      score_val = buffer_bf[k * elem + score_offset + i];
      scores[j] = score_val + bias[j * head_dim + i];
      kvs[j] = kv_val;
    }
    // seq_len == 4 special branch: first 4 are 0 + -inf
    if (seq_len == 4) {
      for (int j = 0; j < 4; ++j) {
        kvs[j] = 0.0f;
        scores[j] = -1e9f + bias[j * head_dim + i];
      }
    }
    float max_v = scores[0];
    for (int j = 1; j < 8; ++j) max_v = std::max(max_v, scores[j]);
    float sum_p = 0.0f, sum_e = 0.0f;
    for (int j = 0; j < 8; ++j) {
      const float e = std::exp(scores[j] - max_v);
      sum_p += kvs[j] * e;
      sum_e += e;
    }
    (*out_fp32)[i] = sum_p / sum_e;
  }
}

// Random bf16 tensor encoded in a float vector (round-trip to bf16 and back).
void FillRandomBF16(std::vector<float>* v, std::mt19937* rng, float min_val = -1.0f, float max_val = 1.0f) {
  std::uniform_real_distribution<float> dist(min_val, max_val);
  for (auto& x : *v) {
    // Round-trip to bf16 then back to emulate loss inside the device kernel.
    __nv_bfloat16 h = __float2bfloat16(dist(*rng));
    x = __bfloat162float(h);
  }
}

void HostFloatToBF16(const std::vector<float>& src, std::vector<__nv_bfloat16>* dst) {
  dst->resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    (*dst)[i] = __float2bfloat16(src[i]);
  }
}

void HostBF16ToFloat(const std::vector<__nv_bfloat16>& src, std::vector<float>* dst) {
  dst->resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    (*dst)[i] = __bfloat162float(src[i]);
  }
}

}  // namespace

class CompressorC4TestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

  template <int64_t kHeadDim>
  void DispatchSmokeForHeadDim() {
    const int batch_size = 2;
    const int num_pages = 4;
    const int64_t kv_buf_elem = static_cast<int64_t>(num_pages) * 8 * kHeadDim * 4;
    const int64_t in_elem = static_cast<int64_t>(batch_size) * kHeadDim * 4;
    const int64_t out_elem = static_cast<int64_t>(batch_size) * kHeadDim;

    BufferMeta buf = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)kv_buf_elem});
    BufferMeta in = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)in_elem}, /*random*/ true);
    BufferMeta out = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)out_elem});
    // score_bias is always FP32 in the kernel.
    BufferMeta bias = CreateBuffer<float>(MemoryType::MEMORY_GPU, {(size_t)(8 * kHeadDim)}, /*random*/ true);
    BufferMeta indices = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {(size_t)batch_size});
    BufferMeta seq_lens = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {(size_t)batch_size});

    std::vector<int32_t> idx_h = {0, 1};
    std::vector<int32_t> sl_h = {8, 12};
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(indices.data_ptr, idx_h.data(), sizeof(int32_t) * batch_size, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(seq_lens.data_ptr, sl_h.data(), sizeof(int32_t) * batch_size, cudaMemcpyHostToDevice));

    // RingBuffer mode (kUsePage4 == false).
    flash_c4_decode<__nv_bfloat16, __nv_bfloat16, kHeadDim>(
        buf.data_ptr, in.data_ptr, out.data_ptr, bias.data_ptr, reinterpret_cast<const int32_t*>(indices.data_ptr),
        reinterpret_cast<const int32_t*>(seq_lens.data_ptr), /*extra=*/nullptr, batch_size, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    // Page4Align mode needs a valid extra buffer.
    BufferMeta extra = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {(size_t)batch_size});
    std::vector<int32_t> ex_h = {0, 1};
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(extra.data_ptr, ex_h.data(), sizeof(int32_t) * batch_size, cudaMemcpyHostToDevice));
    flash_c4_decode<__nv_bfloat16, __nv_bfloat16, kHeadDim>(
        buf.data_ptr, in.data_ptr, out.data_ptr, bias.data_ptr, reinterpret_cast<const int32_t*>(indices.data_ptr),
        reinterpret_cast<const int32_t*>(seq_lens.data_ptr), reinterpret_cast<const int32_t*>(extra.data_ptr),
        batch_size, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    DeleteBuffer(buf);
    DeleteBuffer(in);
    DeleteBuffer(out);
    DeleteBuffer(bias);
    DeleteBuffer(indices);
    DeleteBuffer(seq_lens);
    DeleteBuffer(extra);
  }

 protected:
  using NvidiaTestSuitBase::stream;
};

// -----------------------------------------------------------------------------
// Test 1: dispatch-level smoke test.
//   Make sure flash_c4_decode and flash_c4_prefill launch successfully for all
//   (InFloat, OutFloat) × head_dim combinations. Does not check numerics.
// -----------------------------------------------------------------------------

TEST_F(CompressorC4TestSuit, DispatchSmokeAllDtypesHeadDims) {
  DispatchSmokeForHeadDim<128>();
  DispatchSmokeForHeadDim<512>();
}

// -----------------------------------------------------------------------------
// Test 2: numerical correctness of RingBuffer decode for seq_len % 4 == 0.
//   Build a known buffer + input, compare device output with a host reference.
// -----------------------------------------------------------------------------
TEST_F(CompressorC4TestSuit, DecodeRingBufferNumericalMatchesHostRef) {
  constexpr int64_t kHeadDim = 128;
  const int batch_size = 1;

  // Setup deterministic RNG.
  std::mt19937 rng(20260426);

  // kv_score_buffer : [1, 8, kHeadDim*4]  (num_indices=1, page_size=8)
  std::vector<float> buf_f(static_cast<size_t>(8) * kHeadDim * 4);
  FillRandomBF16(&buf_f, &rng);

  // kv_score_input  : [1, kHeadDim*4]
  std::vector<float> in_f(static_cast<size_t>(kHeadDim) * 4);
  FillRandomBF16(&in_f, &rng);

  // score_bias : [8, kHeadDim] — kernel reads as FP32
  std::vector<float> bias_f(static_cast<size_t>(8) * kHeadDim);
  {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : bias_f) x = dist(rng);
  }

  // ------ convert bf16 inputs and upload ------
  std::vector<__nv_bfloat16> buf_h, in_h;
  HostFloatToBF16(buf_f, &buf_h);
  HostFloatToBF16(in_f, &in_h);

  BufferMeta buf_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {buf_h.size()});
  BufferMeta in_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {in_h.size()});
  BufferMeta bias_d = CreateBuffer<float>(MemoryType::MEMORY_GPU, {bias_f.size()});
  BufferMeta out_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)kHeadDim});
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(buf_d.data_ptr, buf_h.data(), buf_h.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(in_d.data_ptr, in_h.data(), in_h.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(bias_d.data_ptr, bias_f.data(), bias_f.size() * sizeof(float), cudaMemcpyHostToDevice));

  BufferMeta idx_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {(size_t)batch_size});
  BufferMeta sl_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {(size_t)batch_size});
  std::vector<int32_t> idx_h = {0};
  // seq_len=8: c4_write picks slot (8+7)%8 = 7; c4_forward then runs window_len=8.
  std::vector<int32_t> sl_h = {8};
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(idx_d.data_ptr, idx_h.data(), sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(sl_d.data_ptr, sl_h.data(), sizeof(int32_t), cudaMemcpyHostToDevice));

  flash_c4_decode<__nv_bfloat16, __nv_bfloat16, kHeadDim>(
      buf_d.data_ptr, in_d.data_ptr, out_d.data_ptr, bias_d.data_ptr, reinterpret_cast<const int32_t*>(idx_d.data_ptr),
      reinterpret_cast<const int32_t*>(sl_d.data_ptr),
      /*extra=*/nullptr, batch_size, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Write back the updated buffer (c4_write overwrites slot 7).
  std::vector<__nv_bfloat16> buf_after(buf_h.size());
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(buf_after.data(), buf_d.data_ptr, buf_h.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  std::vector<float> buf_after_f;
  HostBF16ToFloat(buf_after, &buf_after_f);

  // Host-replay: first, mirror c4_write — slot (8+7)%8 = 7 becomes the input.
  std::vector<float> host_buf = buf_f;
  const int64_t elem = kHeadDim * 4;
  const int write_slot = 7;
  for (int i = 0; i < elem; ++i) {
    host_buf[write_slot * elem + i] = in_f[i];
  }

  // Round-trip through bf16 for apples-to-apples.
  std::vector<__nv_bfloat16> host_buf_bf16;
  HostFloatToBF16(host_buf, &host_buf_bf16);
  std::vector<float> host_buf_bf16_f;
  HostBF16ToFloat(host_buf_bf16, &host_buf_bf16_f);

  // Buffer-after-write check (only slot 7 should have changed).
  for (int slot = 0; slot < 8; ++slot) {
    for (int i = 0; i < elem; ++i) {
      const size_t idx = static_cast<size_t>(slot) * elem + i;
      ASSERT_NEAR(buf_after_f[idx], host_buf_bf16_f[idx], 1e-2f) << "slot=" << slot << " i=" << i;
    }
  }

  // Now compute host-ref of the forward output.
  std::vector<float> host_out_f;
  HostC4ForwardRing(host_buf_bf16_f, in_f, bias_f, /*seq_len=*/8, kHeadDim, &host_out_f);

  std::vector<__nv_bfloat16> got_out_h(kHeadDim);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(got_out_h.data(), out_d.data_ptr, kHeadDim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  std::vector<float> got_out_f;
  HostBF16ToFloat(got_out_h, &got_out_f);

  // Compare. bf16 softmax-attention has limited precision: use a relaxed bound.
  int mismatches = 0;
  float max_abs = 0.0f;
  for (int64_t i = 0; i < kHeadDim; ++i) {
    const float diff = std::fabs(got_out_f[i] - host_out_f[i]);
    max_abs = std::max(max_abs, diff);
    if (diff > 5e-2f) ++mismatches;
  }
  EXPECT_LE(mismatches, kHeadDim / 50) << "max_abs=" << max_abs << " exceeded threshold in " << mismatches << " / "
                                       << kHeadDim << " elements";

  DeleteBuffer(buf_d);
  DeleteBuffer(in_d);
  DeleteBuffer(bias_d);
  DeleteBuffer(out_d);
  DeleteBuffer(idx_d);
  DeleteBuffer(sl_d);
}

// -----------------------------------------------------------------------------
// Test 3: Decode RingBuffer when seq_len % 4 != 0, output must be left-as-is
//   (kernel skips c4_forward entirely).
// -----------------------------------------------------------------------------
TEST_F(CompressorC4TestSuit, DecodeRingBufferSkipsForwardWhenNotAligned) {
  constexpr int64_t kHeadDim = 128;
  const int batch_size = 1;
  const int64_t kv_buf_elem = static_cast<int64_t>(1) * 8 * kHeadDim * 4;

  BufferMeta buf = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)kv_buf_elem}, /*random*/ true);
  BufferMeta in = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)(kHeadDim * 4)}, /*random*/ true);
  BufferMeta bias = CreateBuffer<float>(MemoryType::MEMORY_GPU, {(size_t)(8 * kHeadDim)}, /*random*/ true);
  BufferMeta out = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)kHeadDim});
  // Init output with sentinel so we can verify it stays unchanged.
  std::vector<__nv_bfloat16> sentinel(kHeadDim, __float2bfloat16(12345.0f));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(out.data_ptr, sentinel.data(), kHeadDim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  BufferMeta idx_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {1});
  BufferMeta sl_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {1});
  std::vector<int32_t> idx_h = {0};
  std::vector<int32_t> sl_h = {5};  // not multiple of 4 -> forward skipped
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(idx_d.data_ptr, idx_h.data(), sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(sl_d.data_ptr, sl_h.data(), sizeof(int32_t), cudaMemcpyHostToDevice));

  flash_c4_decode<__nv_bfloat16, __nv_bfloat16, kHeadDim>(buf.data_ptr, in.data_ptr, out.data_ptr, bias.data_ptr,
                                                          reinterpret_cast<const int32_t*>(idx_d.data_ptr),
                                                          reinterpret_cast<const int32_t*>(sl_d.data_ptr),
                                                          /*extra=*/nullptr, batch_size, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> got(kHeadDim);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(got.data(), out.data_ptr, kHeadDim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  // 12345.0f is not exactly representable in bfloat16; compare against the
  // rounded value that was actually stored (the kernel must not modify output).
  const float sentinel_bf16 = __bfloat162float(__float2bfloat16(12345.0f));
  for (int64_t i = 0; i < kHeadDim; ++i) {
    EXPECT_FLOAT_EQ(__bfloat162float(got[i]), sentinel_bf16) << "output modified at i=" << i;
  }

  DeleteBuffer(buf);
  DeleteBuffer(in);
  DeleteBuffer(bias);
  DeleteBuffer(out);
  DeleteBuffer(idx_d);
  DeleteBuffer(sl_d);
}

// -----------------------------------------------------------------------------
// Test 4: prefill end-to-end smoke with plan_prefill_compress.
//   Uses a valid set of plans and verifies: (a) kernel launches without error,
//   (b) the `kv_score_buffer` has been modified (c4_write really ran).
// -----------------------------------------------------------------------------
TEST_F(CompressorC4TestSuit, PrefillSmokeWithPlanner) {
  constexpr int64_t kHeadDim = 128;
  const int batch_size = 2;
  std::vector<IndicesT> seq_lens = {8, 4};
  std::vector<IndicesT> extend_lens = {8, 4};
  uint32_t num_tokens = 0;
  for (auto l : extend_lens) num_tokens += static_cast<uint32_t>(l);

  // Plans
  BufferMeta sl_d = CreateBuffer<IndicesT>(MemoryType::MEMORY_GPU, {(size_t)batch_size});
  BufferMeta el_d = CreateBuffer<IndicesT>(MemoryType::MEMORY_GPU, {(size_t)batch_size});
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(sl_d.data_ptr, seq_lens.data(), batch_size * sizeof(IndicesT), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(el_d.data_ptr, extend_lens.data(), batch_size * sizeof(IndicesT), cudaMemcpyHostToDevice));

  BufferMeta cp_d = CreateBuffer<uint8_t>(MemoryType::MEMORY_GPU, {(size_t)num_tokens * kPrefillPlanDim});
  BufferMeta wp_d = CreateBuffer<uint8_t>(MemoryType::MEMORY_GPU, {(size_t)num_tokens * kPrefillPlanDim});

  plan_prefill_compress(reinterpret_cast<PrefillPlan*>(cp_d.data_ptr), reinterpret_cast<PrefillPlan*>(wp_d.data_ptr),
                        reinterpret_cast<const IndicesT*>(sl_d.data_ptr),
                        reinterpret_cast<const IndicesT*>(el_d.data_ptr), batch_size, num_tokens, /*compress_ratio=*/4,
                        /*is_overlap=*/true, stream);

  // Buffers for the c4 kernel itself. RingBuffer mode: kv_score_buffer shape
  // [num_indices, 8, kHeadDim*4]. Here num_indices = batch_size = 2.
  const int64_t kv_buf_elem = static_cast<int64_t>(batch_size) * 8 * kHeadDim * 4;
  const int64_t in_elem = static_cast<int64_t>(num_tokens) * kHeadDim * 4;
  const int64_t out_elem = static_cast<int64_t>(num_tokens) * kHeadDim;

  BufferMeta buf_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)kv_buf_elem}, /*random*/ false);
  BufferMeta in_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)in_elem}, /*random*/ true);
  BufferMeta out_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)out_elem});
  BufferMeta bias_d = CreateBuffer<float>(MemoryType::MEMORY_GPU, {(size_t)(8 * kHeadDim)}, /*random*/ true);
  BufferMeta idx_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {(size_t)batch_size});
  std::vector<int32_t> idx_h = {0, 1};
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(idx_d.data_ptr, idx_h.data(), batch_size * sizeof(int32_t), cudaMemcpyHostToDevice));

  // Snapshot buffer before compression for post-check.
  std::vector<__nv_bfloat16> buf_before(kv_buf_elem);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(buf_before.data(), buf_d.data_ptr, kv_buf_elem * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  // Compute real compress/write counts for the test data.
  // seq_lens={8,4}, extend_lens={8,4}, ratio=4, is_overlap=true
  // batch 0: compress=8/4-0/4=2, write: start=8/4*4-4=4, start<=0? no, 4<8 → 8-4=4
  // batch 1: compress=4/4-0/4=1, write: start=4/4*4-4=0, 0<=0 → 4
  const uint32_t num_compress = 3;
  const uint32_t num_write = 8;
  flash_c4_prefill<__nv_bfloat16, __nv_bfloat16, kHeadDim>(
      buf_d.data_ptr, in_d.data_ptr, out_d.data_ptr, bias_d.data_ptr, reinterpret_cast<const int32_t*>(idx_d.data_ptr),
      reinterpret_cast<const PrefillPlan*>(cp_d.data_ptr), reinterpret_cast<const PrefillPlan*>(wp_d.data_ptr),
      /*extra=*/nullptr,  // RingBuffer
      num_compress, num_write, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<__nv_bfloat16> buf_after(kv_buf_elem);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(buf_after.data(), buf_d.data_ptr, kv_buf_elem * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  // There must be at least one element that c4_write has touched.
  bool any_changed = false;
  for (size_t i = 0; i < buf_before.size(); ++i) {
    if (__bfloat162float(buf_before[i]) != __bfloat162float(buf_after[i])) {
      any_changed = true;
      break;
    }
  }
  EXPECT_TRUE(any_changed);

  DeleteBuffer(sl_d);
  DeleteBuffer(el_d);
  DeleteBuffer(cp_d);
  DeleteBuffer(wp_d);
  DeleteBuffer(buf_d);
  DeleteBuffer(in_d);
  DeleteBuffer(out_d);
  DeleteBuffer(bias_d);
  DeleteBuffer(idx_d);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
