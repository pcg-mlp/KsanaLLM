/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * gtest for swa_topk_idxs (BATCHED interface):
 *   - Numerical correctness vs host reference (pure accuracy; no timing
 *     code). Host reference is a verbatim translation of
 *     Dsv4MqaMeta::GetSwaTopkIdxs in dsv4_mqa_meta.cpp, applied per
 *     request and concatenated. The launch + CHECK also covers the launch
 *     smoke role.
 *   - PerfBenchmark_* (env-gated): skipped by default, runs warmup+iters
 *     cudaEvent timing when KLLM_KERNEL_PERF=1.
 *   - Edge cases (req_num=0); no timing.
 */

#include "csrc/kernels/nvidia/swa_topk_idxs/swa_topk_idxs.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

namespace {

constexpr int kDefaultSwaTopkSwaSize = 128;
// -----------------------------------------------------------------------------
// Host reference for a single request — verbatim translation of
//   Dsv4MqaMeta::GetSwaTopkIdxs (dsv4_mqa_meta.cpp), generalised to
//   take the sliding-window size at runtime (the kernel itself now always
//   takes kSwa as a runtime argument).
// -----------------------------------------------------------------------------
void HostRefOneReq(const std::vector<int>& block_ids, int prefix_len, int seq_len, int sliding_window_size,
                   int block_token_num, std::vector<int>* host) {
  const int kSwaSize = sliding_window_size;
  for (int i = prefix_len; i < seq_len; ++i) {
    // NOTE: inclusive upper bound (j <= i) — verbatim match with the CPU
    // reference in dsv4_mqa_meta.cpp. Each row produces kSwaSize outputs.
    for (int j = i - kSwaSize + 1; j <= i; ++j) {
      if (j < 0) {
        host->push_back(-1);
      } else {
        host->push_back(block_ids[j / block_token_num] * block_token_num + j % block_token_num);
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Holds one request's host + device buffers and wires them into the batched
// argument arrays. Constructor allocates d_block_ids and copies data up.
// Destructor frees. Keep all BatchedRequests alive until the kernel launch
// completes.
// -----------------------------------------------------------------------------
struct ReqBuf {
  std::vector<int> h_block_ids;
  int prefix_len = 0;
  int seq_len = 0;
  int* d_block_ids = nullptr;

  ReqBuf() = default;
  ReqBuf(const ReqBuf&) = delete;
  ReqBuf& operator=(const ReqBuf&) = delete;
  ~ReqBuf() {
    if (d_block_ids) cudaFree(d_block_ids);
  }

  void Init(std::vector<int> ids, int pref, int seq) {
    h_block_ids = std::move(ids);
    prefix_len = pref;
    seq_len = seq;
    if (!h_block_ids.empty()) {
      CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_block_ids, sizeof(int) * h_block_ids.size()));
      CHECK_NVIDIA_CUDA_ERROR(
          cudaMemcpy(d_block_ids, h_block_ids.data(), sizeof(int) * h_block_ids.size(), cudaMemcpyHostToDevice));
    }
  }

  int ExtendLen() const { return seq_len - prefix_len; }
};

// Packs a list of ReqBuf into the batched device-side argument arrays the
// kernel expects. Returned BatchBuf owns the four device arrays.
struct BatchBuf {
  const int** d_block_ids_array = nullptr;  // const int* const* on device
  int* d_prefix_lens = nullptr;
  int* d_seq_lens = nullptr;
  int* d_out_offsets = nullptr;
  int* d_out = nullptr;
  size_t out_count = 0;  // in ints

  BatchBuf() = default;
  BatchBuf(const BatchBuf&) = delete;
  BatchBuf& operator=(const BatchBuf&) = delete;
  ~BatchBuf() {
    if (d_block_ids_array) cudaFree(d_block_ids_array);
    if (d_prefix_lens) cudaFree(d_prefix_lens);
    if (d_seq_lens) cudaFree(d_seq_lens);
    if (d_out_offsets) cudaFree(d_out_offsets);
    if (d_out) cudaFree(d_out);
  }

  void Pack(const std::vector<ReqBuf*>& reqs, int sliding_window_size) {
    const int req_num = static_cast<int>(reqs.size());
    std::vector<const int*> h_ptrs(req_num);
    std::vector<int> h_prefix(req_num), h_seq(req_num), h_offsets(req_num);
    int row_cursor = 0;
    for (int r = 0; r < req_num; ++r) {
      h_ptrs[r] = reqs[r]->d_block_ids;
      h_prefix[r] = reqs[r]->prefix_len;
      h_seq[r] = reqs[r]->seq_len;
      h_offsets[r] = row_cursor;
      row_cursor += reqs[r]->ExtendLen();
    }
    const int kK = sliding_window_size;
    out_count = static_cast<size_t>(row_cursor) * static_cast<size_t>(kK);

    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_block_ids_array, sizeof(const int*) * req_num));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_block_ids_array, h_ptrs.data(), sizeof(const int*) * req_num, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_prefix_lens, sizeof(int) * req_num));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_prefix_lens, h_prefix.data(), sizeof(int) * req_num, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_seq_lens, sizeof(int) * req_num));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_seq_lens, h_seq.data(), sizeof(int) * req_num, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_out_offsets, sizeof(int) * req_num));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_out_offsets, h_offsets.data(), sizeof(int) * req_num, cudaMemcpyHostToDevice));
    if (out_count > 0) {
      CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_out, sizeof(int) * out_count));
      // Sentinel fill to catch untouched output slots.
      CHECK_NVIDIA_CUDA_ERROR(cudaMemset(d_out, 0xAB, sizeof(int) * out_count));
    }
  }
};

// Return true when env KLLM_KERNEL_PERF is set to a truthy value
// ("1" / "true" / non-zero). Cached once per process.
inline bool KernelPerfEnabled() {
  static const bool kEnabled = []() {
    const char* v = std::getenv("KLLM_KERNEL_PERF");
    if (v == nullptr) return false;
    const std::string s(v);
    return !(s.empty() || s == "0" || s == "false" || s == "False" || s == "FALSE");
  }();
  return kEnabled;
}

}  // namespace

class NvidiaSwaTopkIdxsTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
};

// -----------------------------------------------------------------------------
// Test 1: Numerical correctness — single request, context phase.
// Validates the kernel still matches the CPU reference when req_num == 1.
// -----------------------------------------------------------------------------
TEST_F(NvidiaSwaTopkIdxsTestSuit, NumericalMatchesHostRef_SingleContext) {
  constexpr int kPrefixLen = 0;
  constexpr int kSeqLen = 200;
  constexpr int kBlockTokenNum = 16;
  constexpr int kSwa = 128;
  constexpr int kNumBlocks = (kSeqLen + kBlockTokenNum - 1) / kBlockTokenNum;

  std::mt19937 rng(0x13579u);
  std::uniform_int_distribution<int> dist(0, 100000);
  std::vector<int> ids(kNumBlocks);
  for (auto& v : ids) v = dist(rng);

  std::vector<int> h_ref;
  HostRefOneReq(ids, kPrefixLen, kSeqLen, kSwa, kBlockTokenNum, &h_ref);

  ReqBuf req;
  req.Init(std::move(ids), kPrefixLen, kSeqLen);
  BatchBuf batch;
  batch.Pack({&req}, kSwa);

  InvokeSwaTopkIdxs(batch.d_block_ids_array, batch.d_prefix_lens, batch.d_seq_lens, batch.d_out,
                         batch.d_out_offsets,
                         /*req_num=*/1, kSwa, kBlockTokenNum, stream);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<int> h_out(batch.out_count);
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_out.data(), batch.d_out, sizeof(int) * batch.out_count, cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_out.size(), h_ref.size());
  for (size_t idx = 0; idx < h_out.size(); ++idx) {
    EXPECT_EQ(h_out[idx], h_ref[idx]) << "mismatch at idx=" << idx;
  }
}

// -----------------------------------------------------------------------------
// Test 2: Numerical correctness — BATCHED mixed context + decode.
//   req0: prefix=0, seq=200   (context, lots of j<0 sentinels)
//   req1: prefix=1024, seq=1025 (decode, extend_len=1)
//   req2: prefix=50,  seq=90  (mid-range)
//   req3: prefix=0,   seq=1   (tiny, extend_len=1, mostly sentinels)
// -----------------------------------------------------------------------------
TEST_F(NvidiaSwaTopkIdxsTestSuit, NumericalMatchesHostRef_BatchedMixed) {
  constexpr int kBlockTokenNum = 16;
  constexpr int kSwa = 128;

  struct ReqSpec {
    int prefix_len;
    int seq_len;
    uint32_t seed;
  };
  const std::vector<ReqSpec> specs = {
      {0, 200, 0xAAAAu},
      {1024, 1025, 0xBBBBu},
      {50, 90, 0xCCCCu},
      {0, 1, 0xDDDDu},
  };

  std::vector<std::unique_ptr<ReqBuf>> reqs;
  std::vector<ReqBuf*> req_ptrs;
  std::vector<int> h_ref;

  for (const auto& s : specs) {
    std::mt19937 rng(s.seed);
    std::uniform_int_distribution<int> dist(0, 100000);
    const int num_blocks = (s.seq_len + kBlockTokenNum - 1) / kBlockTokenNum;
    std::vector<int> ids(std::max(1, num_blocks));
    for (auto& v : ids) v = dist(rng);

    HostRefOneReq(ids, s.prefix_len, s.seq_len, kSwa, kBlockTokenNum, &h_ref);

    auto buf = std::make_unique<ReqBuf>();
    buf->Init(std::move(ids), s.prefix_len, s.seq_len);
    req_ptrs.push_back(buf.get());
    reqs.push_back(std::move(buf));
  }

  BatchBuf batch;
  batch.Pack(req_ptrs, kSwa);

  InvokeSwaTopkIdxs(batch.d_block_ids_array, batch.d_prefix_lens, batch.d_seq_lens, batch.d_out,
                         batch.d_out_offsets,
                         /*req_num=*/static_cast<int>(specs.size()), kSwa, kBlockTokenNum, stream);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<int> h_out(batch.out_count);
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_out.data(), batch.d_out, sizeof(int) * batch.out_count, cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_out.size(), h_ref.size());
  for (size_t idx = 0; idx < h_out.size(); ++idx) {
    EXPECT_EQ(h_out[idx], h_ref[idx]) << "mismatch at idx=" << idx;
  }
}

// -----------------------------------------------------------------------------
// Test 3: Numerical correctness — BATCHED with one request having
// extend_len == 0 (must be a no-op for that request; neighbours unaffected).
// -----------------------------------------------------------------------------
TEST_F(NvidiaSwaTopkIdxsTestSuit, NumericalMatchesHostRef_BatchedWithEmptyReq) {
  constexpr int kBlockTokenNum = 32;
  constexpr int kSwa = 128;

  std::mt19937 rng(0xBEEFu);
  std::uniform_int_distribution<int> dist(0, 100000);

  ReqBuf r0, r1, r2;
  // r0: normal context
  {
    std::vector<int> ids(16);
    for (auto& v : ids) v = dist(rng);
    r0.Init(std::move(ids), 0, 300);
  }
  // r1: empty extend (prefix == seq)
  {
    std::vector<int> ids(8);
    for (auto& v : ids) v = dist(rng);
    r1.Init(std::move(ids), 128, 128);
  }
  // r2: decode
  {
    std::vector<int> ids(32);
    for (auto& v : ids) v = dist(rng);
    r2.Init(std::move(ids), 1000, 1001);
  }

  std::vector<int> h_ref;
  HostRefOneReq(r0.h_block_ids, r0.prefix_len, r0.seq_len, kSwa, kBlockTokenNum, &h_ref);
  HostRefOneReq(r1.h_block_ids, r1.prefix_len, r1.seq_len, kSwa, kBlockTokenNum, &h_ref);  // no-op
  HostRefOneReq(r2.h_block_ids, r2.prefix_len, r2.seq_len, kSwa, kBlockTokenNum, &h_ref);

  BatchBuf batch;
  batch.Pack({&r0, &r1, &r2}, kSwa);

  InvokeSwaTopkIdxs(batch.d_block_ids_array, batch.d_prefix_lens, batch.d_seq_lens, batch.d_out,
                         batch.d_out_offsets, /*req_num=*/3, kSwa, kBlockTokenNum, stream);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<int> h_out(batch.out_count);
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_out.data(), batch.d_out, sizeof(int) * batch.out_count, cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_out.size(), h_ref.size());
  for (size_t idx = 0; idx < h_out.size(); ++idx) {
    EXPECT_EQ(h_out[idx], h_ref[idx]) << "mismatch at idx=" << idx;
  }
}

// -----------------------------------------------------------------------------
// Test 4: Performance benchmark — BATCHED decode (typical serving shape,
// req_num large, each req extend_len == 1). Env-gated.
// -----------------------------------------------------------------------------
TEST_F(NvidiaSwaTopkIdxsTestSuit, PerfBenchmark_BatchedDecode) {
  if (!KernelPerfEnabled()) {
    GTEST_SKIP() << "perf disabled; set KLLM_KERNEL_PERF=1 to enable";
  }
  constexpr int kWarmup = 5;
  constexpr int kIters = 200;
  constexpr int kReqNum = 64;
  constexpr int kPrefixLen = 1024;
  constexpr int kSeqLen = 1025;
  constexpr int kBlockTokenNum = 32;
  constexpr int kNumBlocks = (kSeqLen + kBlockTokenNum - 1) / kBlockTokenNum;

  std::vector<std::unique_ptr<ReqBuf>> reqs;
  std::vector<ReqBuf*> req_ptrs;
  std::mt19937 rng(0xD00Du);
  std::uniform_int_distribution<int> dist(0, 100000);
  for (int r = 0; r < kReqNum; ++r) {
    std::vector<int> ids(kNumBlocks);
    for (auto& v : ids) v = dist(rng);
    auto buf = std::make_unique<ReqBuf>();
    buf->Init(std::move(ids), kPrefixLen, kSeqLen);
    req_ptrs.push_back(buf.get());
    reqs.push_back(std::move(buf));
  }
  BatchBuf batch;
  batch.Pack(req_ptrs, kDefaultSwaTopkSwaSize);

  for (int i = 0; i < kWarmup; ++i) {
    InvokeSwaTopkIdxs(batch.d_block_ids_array, batch.d_prefix_lens, batch.d_seq_lens, batch.d_out,
                           batch.d_out_offsets, kReqNum, kDefaultSwaTopkSwaSize, kBlockTokenNum, stream);
  }
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  cudaEvent_t ev_begin{}, ev_end{};
  CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&ev_begin));
  CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&ev_end));
  CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(ev_begin, stream));
  for (int i = 0; i < kIters; ++i) {
    InvokeSwaTopkIdxs(batch.d_block_ids_array, batch.d_prefix_lens, batch.d_seq_lens, batch.d_out,
                           batch.d_out_offsets, kReqNum, kDefaultSwaTopkSwaSize, kBlockTokenNum, stream);
  }
  CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(ev_end, stream));
  CHECK_NVIDIA_CUDA_ERROR(cudaEventSynchronize(ev_end));

  float ms_total = 0.0f;
  CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&ms_total, ev_begin, ev_end));
  std::cout << "[Perf] swa_topk_idxs BatchedDecode"
            << " req_num=" << kReqNum << " prefix=" << kPrefixLen << " seq=" << kSeqLen << " avg=" << ms_total / kIters
            << " ms iters=" << kIters << std::endl;
  CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(ev_begin));
  CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(ev_end));

  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// -----------------------------------------------------------------------------
// Test 6: Edge — req_num == 0 is a pure no-op; nullptrs OK.
// -----------------------------------------------------------------------------
TEST_F(NvidiaSwaTopkIdxsTestSuit, EdgeEmptyBatch) {
  InvokeSwaTopkIdxs(/*block_ids_array=*/nullptr, /*prefix_lens=*/nullptr,
                         /*seq_lens=*/nullptr, /*out=*/nullptr, /*out_offsets=*/nullptr,
                         /*req_num=*/0, /*sliding_window_size=*/kDefaultSwaTopkSwaSize,
                         /*block_token_num=*/16, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
