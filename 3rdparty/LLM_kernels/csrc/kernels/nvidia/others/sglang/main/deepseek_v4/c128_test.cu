/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Unit tests for the C128 KV-compression kernel ported from SGLang DeepSeek-V4:
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/c128.cuh
 */

#include "csrc/kernels/nvidia/others/sglang/main/deepseek_v4/c128.cuh"

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


// C128 ring buffer depth
static constexpr int kC128RingDepth = 128;

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

// CPU reference for c128_forward (decode path, window_len=128).
//
// buffer_f: [128, head_dim*2]  layout per slot: | kv (head_dim) | score (head_dim) |
// score_bias: [128, head_dim]
// out_f: [head_dim]
//
// Computes online softmax:
//   score[j][i]   = buffer_f[j*head_dim*2 + head_dim + i] + score_bias[j*head_dim + i]
//   kv[j][i]      = buffer_f[j*head_dim*2 + i]
//   out[i]        = sum_j(kv[j][i] * exp(score[j][i] - max)) / sum_j(exp(score[j][i] - max))
void HostC128Forward(const std::vector<float>& buffer_f /*[128, head_dim*2]*/,
                     const std::vector<float>& score_bias /*[128, head_dim]*/, int64_t head_dim,
                     std::vector<float>* out_f /*[head_dim]*/) {
  out_f->assign(static_cast<size_t>(head_dim), 0.0f);
  const int64_t elem_size = head_dim * 2;
  for (int64_t i = 0; i < head_dim; ++i) {
    // Find max score
    float max_score = -1e30f;
    for (int j = 0; j < kC128RingDepth; ++j) {
      const float score = buffer_f[j * elem_size + head_dim + i] + score_bias[j * head_dim + i];
      if (score > max_score) max_score = score;
    }
    // Compute softmax weighted sum
    float exp_sum = 0.0f;
    float product = 0.0f;
    for (int j = 0; j < kC128RingDepth; ++j) {
      const float kv_val = buffer_f[j * elem_size + i];
      const float score = buffer_f[j * elem_size + head_dim + i] + score_bias[j * head_dim + i];
      const float e = std::exp(score - max_score);
      product += kv_val * e;
      exp_sum += e;
    }
    (*out_f)[i] = product / exp_sum;
  }
}

}  // namespace

class CompressorC128TestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
};

// -----------------------------------------------------------------------------
// Test 1: dispatch-level smoke test.
//   Launch flash_c128_decode for all (InFloat, OutFloat) × head_dim combinations.
//   seq_len=128 so forward pass is triggered. Does not check numerics.
// -----------------------------------------------------------------------------
TEST_F(CompressorC128TestSuit, DispatchSmokeAllDtypesHeadDims) {
  const uint32_t batch_size = 2;
  const uint32_t num_indices = batch_size;

  for (int head_dim : {128, 256}) {
    const int64_t kv_buf_elem = static_cast<int64_t>(num_indices) * kC128RingDepth * head_dim * 2;
    const int64_t in_elem = static_cast<int64_t>(batch_size) * head_dim * 2;
    const int64_t out_elem = static_cast<int64_t>(batch_size) * head_dim;
    const int64_t bias_elem = static_cast<int64_t>(kC128RingDepth) * head_dim;

    BufferMeta buf = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)kv_buf_elem});
    BufferMeta in = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)in_elem}, /*random*/ true);
    BufferMeta out_bf = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)out_elem});
    BufferMeta out_fp = CreateBuffer<float>(MemoryType::MEMORY_GPU, {(size_t)out_elem});
    BufferMeta bias = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)bias_elem}, /*random*/ true);
    BufferMeta idx_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size});
    BufferMeta sl_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size});

    std::vector<int32_t> idx_h = {0, 1};
    std::vector<int32_t> sl_h = {128, 256};  // multiples of 128 → forward triggered
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(idx_d.data_ptr, idx_h.data(), batch_size * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(sl_d.data_ptr, sl_h.data(), batch_size * sizeof(int32_t), cudaMemcpyHostToDevice));

    // bf16 in → bf16 out
    if (head_dim == 128) {
      flash_c128_decode<__nv_bfloat16, __nv_bfloat16, 128>(
          buf.data_ptr, in.data_ptr, out_bf.data_ptr, bias.data_ptr, reinterpret_cast<const int32_t*>(idx_d.data_ptr),
          reinterpret_cast<const int32_t*>(sl_d.data_ptr), batch_size, stream);
    } else {
      flash_c128_decode<__nv_bfloat16, __nv_bfloat16, 256>(
          buf.data_ptr, in.data_ptr, out_bf.data_ptr, bias.data_ptr, reinterpret_cast<const int32_t*>(idx_d.data_ptr),
          reinterpret_cast<const int32_t*>(sl_d.data_ptr), batch_size, stream);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    // bf16 in → float out
    if (head_dim == 128) {
      flash_c128_decode<__nv_bfloat16, float, 128>(buf.data_ptr, in.data_ptr, out_fp.data_ptr, bias.data_ptr,
                                                   reinterpret_cast<const int32_t*>(idx_d.data_ptr),
                                                   reinterpret_cast<const int32_t*>(sl_d.data_ptr), batch_size, stream);
    } else {
      flash_c128_decode<__nv_bfloat16, float, 256>(buf.data_ptr, in.data_ptr, out_fp.data_ptr, bias.data_ptr,
                                                   reinterpret_cast<const int32_t*>(idx_d.data_ptr),
                                                   reinterpret_cast<const int32_t*>(sl_d.data_ptr), batch_size, stream);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    // float in, float buf → float out
    {
      BufferMeta buf_fp = CreateBuffer<float>(MemoryType::MEMORY_GPU, {(size_t)kv_buf_elem});
      BufferMeta in_fp = CreateBuffer<float>(MemoryType::MEMORY_GPU, {(size_t)in_elem}, /*random*/ true);
      BufferMeta bias_fp = CreateBuffer<float>(MemoryType::MEMORY_GPU, {(size_t)bias_elem}, /*random*/ true);

      if (head_dim == 128) {
        flash_c128_decode<float, float, 128>(buf_fp.data_ptr, in_fp.data_ptr, out_fp.data_ptr, bias_fp.data_ptr,
                                             reinterpret_cast<const int32_t*>(idx_d.data_ptr),
                                             reinterpret_cast<const int32_t*>(sl_d.data_ptr), batch_size, stream);
      } else {
        flash_c128_decode<float, float, 256>(buf_fp.data_ptr, in_fp.data_ptr, out_fp.data_ptr, bias_fp.data_ptr,
                                             reinterpret_cast<const int32_t*>(idx_d.data_ptr),
                                             reinterpret_cast<const int32_t*>(sl_d.data_ptr), batch_size, stream);
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      EXPECT_EQ(cudaGetLastError(), cudaSuccess);

      DeleteBuffer(buf_fp);
      DeleteBuffer(in_fp);
      DeleteBuffer(bias_fp);
    }

    DeleteBuffer(buf);
    DeleteBuffer(in);
    DeleteBuffer(out_bf);
    DeleteBuffer(out_fp);
    DeleteBuffer(bias);
    DeleteBuffer(idx_d);
    DeleteBuffer(sl_d);
  }
}

// -----------------------------------------------------------------------------
// Test 2: numerical correctness for bf16 decode, head_dim=128.
//   Build a deterministic ring buffer, call flash_c128_decode with seq_len=128,
//   compare device output with the CPU reference HostC128Forward.
// -----------------------------------------------------------------------------
TEST_F(CompressorC128TestSuit, DecodeNumericalMatchesBF16HeadDim128) {
  constexpr int64_t head_dim = 128;
  const uint32_t batch_size = 1;
  const int32_t seq_len = 128;  // triggers c128_forward

  std::mt19937 rng(20260426u);

  // kv_score_buffer: [1, 128, head_dim*2]
  const size_t buf_elem = static_cast<size_t>(kC128RingDepth) * head_dim * 2;
  // kv_score_input: [1, head_dim*2]
  const size_t in_elem = static_cast<size_t>(head_dim) * 2;
  // score_bias: [128, head_dim]
  const size_t bias_elem = static_cast<size_t>(kC128RingDepth) * head_dim;

  std::vector<float> buf_f(buf_elem), in_f(in_elem), bias_f(bias_elem);
  FillRandomBF16(&buf_f, &rng);
  FillRandomBF16(&in_f, &rng);
  FillRandomBF16(&bias_f, &rng);

  std::vector<__nv_bfloat16> buf_h, in_h, bias_h;
  HostFloatToBF16(buf_f, &buf_h);
  HostFloatToBF16(in_f, &in_h);
  HostFloatToBF16(bias_f, &bias_h);
  // Recover exact bf16-rounded values for CPU reference
  HostBF16ToFloat(buf_h, &buf_f);
  HostBF16ToFloat(in_h, &in_f);
  HostBF16ToFloat(bias_h, &bias_f);

  BufferMeta buf_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {buf_elem});
  BufferMeta in_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {in_elem});
  BufferMeta bias_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {bias_elem});
  BufferMeta out_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)head_dim});
  BufferMeta idx_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size});
  BufferMeta sl_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size});

  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(buf_d.data_ptr, buf_h.data(), buf_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(in_d.data_ptr, in_h.data(), in_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(bias_d.data_ptr, bias_h.data(), bias_elem * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  std::vector<int32_t> idx_h2 = {0};
  std::vector<int32_t> sl_h2 = {seq_len};
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(idx_d.data_ptr, idx_h2.data(), sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(sl_d.data_ptr, sl_h2.data(), sizeof(int32_t), cudaMemcpyHostToDevice));

  flash_c128_decode<__nv_bfloat16, __nv_bfloat16, head_dim>(
      buf_d.data_ptr, in_d.data_ptr, out_d.data_ptr, bias_d.data_ptr, reinterpret_cast<const int32_t*>(idx_d.data_ptr),
      reinterpret_cast<const int32_t*>(sl_d.data_ptr), batch_size, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Read device output
  std::vector<__nv_bfloat16> got_h(head_dim);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(got_h.data(), out_d.data_ptr, head_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  std::vector<float> got;
  HostBF16ToFloat(got_h, &got);

  // CPU reference: mirror c128_write at slot (seq_len + 127) % 128 = 127
  std::vector<float> host_buf = buf_f;
  const int write_slot = (seq_len + 127) % kC128RingDepth;  // = 127
  const int64_t elem_size = head_dim * 2;
  for (int64_t i = 0; i < elem_size; ++i) {
    host_buf[static_cast<size_t>(write_slot) * elem_size + i] = in_f[i];
  }
  // Round-trip through bf16
  std::vector<__nv_bfloat16> host_buf_h;
  HostFloatToBF16(host_buf, &host_buf_h);
  HostBF16ToFloat(host_buf_h, &host_buf);

  std::vector<float> ref;
  HostC128Forward(host_buf, bias_f, head_dim, &ref);

  // Compare (bf16 softmax has limited precision)
  int mismatches = 0;
  float max_abs = 0.0f;
  for (int64_t i = 0; i < head_dim; ++i) {
    const float diff = std::fabs(got[i] - ref[i]);
    max_abs = std::max(max_abs, diff);
    if (diff > 5e-2f) ++mismatches;
  }
  EXPECT_LE(mismatches, static_cast<int>(head_dim / 50))
      << "max_abs=" << max_abs << " mismatches=" << mismatches << "/" << head_dim;

  DeleteBuffer(buf_d);
  DeleteBuffer(in_d);
  DeleteBuffer(bias_d);
  DeleteBuffer(out_d);
  DeleteBuffer(idx_d);
  DeleteBuffer(sl_d);
}

// -----------------------------------------------------------------------------
// Test 3: decode skips c128_forward when seq_len % 128 != 0.
//   Output must remain unchanged (initialized with a sentinel value).
// -----------------------------------------------------------------------------
TEST_F(CompressorC128TestSuit, DecodeSkipsForwardWhenNotAligned) {
  constexpr int64_t head_dim = 128;
  const uint32_t batch_size = 1;

  const size_t buf_elem = static_cast<size_t>(kC128RingDepth) * head_dim * 2;
  const size_t in_elem = static_cast<size_t>(head_dim) * 2;
  const size_t bias_elem = static_cast<size_t>(kC128RingDepth) * head_dim;

  BufferMeta buf_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {buf_elem}, /*random*/ true);
  BufferMeta in_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {in_elem}, /*random*/ true);
  BufferMeta bias_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {bias_elem}, /*random*/ true);
  BufferMeta out_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)head_dim});
  BufferMeta idx_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size});
  BufferMeta sl_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {batch_size});

  // Initialize output with a sentinel that is not a multiple of 128
  std::vector<__nv_bfloat16> sentinel(head_dim, __float2bfloat16(12345.0f));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(out_d.data_ptr, sentinel.data(), head_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  std::vector<int32_t> idx_h = {0};
  std::vector<int32_t> sl_h = {5};  // not a multiple of 128 → forward skipped
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(idx_d.data_ptr, idx_h.data(), sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(sl_d.data_ptr, sl_h.data(), sizeof(int32_t), cudaMemcpyHostToDevice));

  flash_c128_decode<__nv_bfloat16, __nv_bfloat16, head_dim>(
      buf_d.data_ptr, in_d.data_ptr, out_d.data_ptr, bias_d.data_ptr, reinterpret_cast<const int32_t*>(idx_d.data_ptr),
      reinterpret_cast<const int32_t*>(sl_d.data_ptr), batch_size, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> got(head_dim);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(got.data(), out_d.data_ptr, head_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  const float expected = __bfloat162float(__float2bfloat16(12345.0f));
  for (int64_t i = 0; i < head_dim; ++i) {
    EXPECT_FLOAT_EQ(__bfloat162float(got[i]), expected) << "output modified at i=" << i;
  }

  DeleteBuffer(buf_d);
  DeleteBuffer(in_d);
  DeleteBuffer(bias_d);
  DeleteBuffer(out_d);
  DeleteBuffer(idx_d);
  DeleteBuffer(sl_d);
}

// -----------------------------------------------------------------------------
// Test 4: prefill end-to-end smoke with plan_prefill_compress (compress_ratio=128).
//   Verifies: (a) kernel launches without error,
//             (b) the kv_score_buffer has been modified (c128_write really ran).
// -----------------------------------------------------------------------------
TEST_F(CompressorC128TestSuit, PrefillSmokeWithPlanner) {
  constexpr int64_t head_dim = 128;
  const int batch_size = 2;
  // seq_lens >= compress_ratio=128 and extend_lens multiples of compress_ratio
  std::vector<IndicesT> seq_lens = {128, 256};
  std::vector<IndicesT> extend_lens = {128, 128};
  uint32_t num_tokens = 0;
  for (auto l : extend_lens) num_tokens += static_cast<uint32_t>(l);

  // Upload seq_lens and extend_lens
  BufferMeta sl_d = CreateBuffer<IndicesT>(MemoryType::MEMORY_GPU, {(size_t)batch_size});
  BufferMeta el_d = CreateBuffer<IndicesT>(MemoryType::MEMORY_GPU, {(size_t)batch_size});
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(sl_d.data_ptr, seq_lens.data(), batch_size * sizeof(IndicesT), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(el_d.data_ptr, extend_lens.data(), batch_size * sizeof(IndicesT), cudaMemcpyHostToDevice));

  // Allocate plan buffers
  BufferMeta cp_d = CreateBuffer<uint8_t>(MemoryType::MEMORY_GPU, {(size_t)num_tokens * kPrefillPlanDim});
  BufferMeta wp_d = CreateBuffer<uint8_t>(MemoryType::MEMORY_GPU, {(size_t)num_tokens * kPrefillPlanDim});

  plan_prefill_compress(reinterpret_cast<PrefillPlan*>(cp_d.data_ptr), reinterpret_cast<PrefillPlan*>(wp_d.data_ptr),
                        reinterpret_cast<const IndicesT*>(sl_d.data_ptr),
                        reinterpret_cast<const IndicesT*>(el_d.data_ptr), batch_size, num_tokens,
                        /*compress_ratio=*/128,
                        /*is_overlap=*/true, stream);

  // kv_score_buffer: [num_indices, 128, head_dim*2]; num_indices = batch_size = 2
  const int64_t kv_buf_elem = static_cast<int64_t>(batch_size) * kC128RingDepth * head_dim * 2;
  const int64_t in_elem = static_cast<int64_t>(num_tokens) * head_dim * 2;
  const int64_t out_elem = static_cast<int64_t>(num_tokens) * head_dim;
  const int64_t bias_elem = static_cast<int64_t>(kC128RingDepth) * head_dim;

  BufferMeta buf_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)kv_buf_elem}, /*random*/ false);
  BufferMeta in_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)in_elem}, /*random*/ true);
  BufferMeta out_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)out_elem});
  BufferMeta bias_d = CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {(size_t)bias_elem}, /*random*/ true);
  BufferMeta idx_d = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {(size_t)batch_size});

  std::vector<int32_t> idx_h = {0, 1};
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(idx_d.data_ptr, idx_h.data(), batch_size * sizeof(int32_t), cudaMemcpyHostToDevice));

  // Snapshot buffer before
  std::vector<__nv_bfloat16> buf_before(kv_buf_elem);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(buf_before.data(), buf_d.data_ptr, kv_buf_elem * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  // Compute real compress/write counts.
  // seq_lens={128,256}, extend_lens={128,128}, ratio=128, is_overlap=true
  // batch 0: compress=128/128-0=1, write: start=128-128=0, 0<=0 → 128
  // batch 1: compress=256/128-128/128=1, write: start=256-128=128, 128<=128 → 128
  const uint32_t num_compress = 2;
  const uint32_t num_write = num_tokens;  // 256
  flash_c128_prefill<__nv_bfloat16, __nv_bfloat16, head_dim>(
      buf_d.data_ptr, in_d.data_ptr, out_d.data_ptr, bias_d.data_ptr, reinterpret_cast<const int32_t*>(idx_d.data_ptr),
      reinterpret_cast<const int32_t*>(idx_d.data_ptr),  // load_indices == indices
      reinterpret_cast<const PrefillPlan*>(cp_d.data_ptr), reinterpret_cast<const PrefillPlan*>(wp_d.data_ptr),
      num_compress, num_write, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  // Verify that at least one element in the buffer was written
  std::vector<__nv_bfloat16> buf_after(kv_buf_elem);
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(buf_after.data(), buf_d.data_ptr, kv_buf_elem * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  bool any_changed = false;
  for (size_t i = 0; i < static_cast<size_t>(kv_buf_elem); ++i) {
    if (__bfloat162float(buf_before[i]) != __bfloat162float(buf_after[i])) {
      any_changed = true;
      break;
    }
  }
  EXPECT_TRUE(any_changed) << "kv_score_buffer should have been modified by c128_write";

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
