/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "csrc/kernels/nvidia/attention_meta/prepare_dsv4_mqa_meta.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class NvidiaPrepareDsv4MqaMetaTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  static constexpr int kTopkBlockSize = 64;  // matches MqaSharedAttentionMeta::kTopkBlockSize
  static constexpr int kSwaSize = 128;       // matches MqaSharedAttentionMeta::kSwaSize

  // Returns -1 when the position falls outside the valid SWA window.
  static int HostSwaPhysPos(const std::vector<int>& swa_block_ids, const std::vector<size_t>& swa_block_idx_offsets,
                            int token_pos, int block_token_num) {
    if (token_pos < 0) return -1;
    if (swa_block_idx_offsets.empty()) return -1;
    const int global_block_idx = token_pos / block_token_num;
    const int local_swa_idx = global_block_idx - static_cast<int>(swa_block_idx_offsets[0]);
    if (local_swa_idx < 0 || local_swa_idx >= static_cast<int>(swa_block_ids.size())) return -1;
    return swa_block_ids[local_swa_idx] * block_token_num + (token_pos % block_token_num);
  }

  // Host reference for c128_block_indices and c128_topk_lengths.
  static void HostRefC128(const std::vector<int32_t>& cu_extend_lens, const std::vector<int32_t>& cu_prefix_lens,
                          const std::vector<int32_t>& block_table, int batch_size, int block_token_num,
                          int max_blocks_per_seq, int aligned_c128_max, bool fill_topk_with_stride,
                          std::vector<int32_t>& c128_block_indices, std::vector<int32_t>& c128_topk_lengths) {
    constexpr int kC128Ratio = 128;
    const int c128_slots = block_token_num / kC128Ratio;
    for (int r = 0; r < batch_size; ++r) {
      const int prefix_len = cu_prefix_lens[r + 1] - cu_prefix_lens[r];
      const int extend_begin = cu_extend_lens[r];
      const int extend_len = cu_extend_lens[r + 1] - extend_begin;
      const int req_topk_len = (prefix_len + extend_len) / kC128Ratio;
      c128_topk_lengths[r] = fill_topk_with_stride ? aligned_c128_max : req_topk_len;
      for (int local = 0; local < extend_len; ++local) {
        const int i = prefix_len + local;
        const int t = extend_begin + local;
        const int c128_len = (i + 1) / kC128Ratio;
        for (int j = 0; j < aligned_c128_max; ++j) {
          int32_t v = -1;
          if (j < c128_len) {
            const int j_token = (j + 1) * kC128Ratio - 1;
            const int j_blk = block_table[r * max_blocks_per_seq + j_token / block_token_num];
            if (j_blk >= 0) {
              v = j_blk * c128_slots + (j_token / kC128Ratio) % c128_slots;
            }
          }
          c128_block_indices[t * aligned_c128_max + j] = v;
        }
      }
    }
  }

  struct ReqDescr {
    int prefix_len;
    int extend_len;
    std::vector<int> swa_block_ids;
    std::vector<size_t> swa_block_idx_offsets;
  };

  // Builds the standard cu_*/block_table/c128 dimensions plus per-req SWA layouts.
  struct BatchSetup {
    std::vector<int32_t> cu_extend_lens;
    std::vector<int32_t> cu_prefix_lens;
    std::vector<int32_t> block_table;
    std::vector<ReqDescr> reqs;
    int num_tokens;
    int max_blocks_per_seq;
    int aligned_c128_max;
    int max_extend_len;
  };

  static BatchSetup BuildBatchSetup(const std::vector<ReqDescr>& reqs, int block_token_num) {
    constexpr int kC128Ratio = 128;
    const int batch_size = static_cast<int>(reqs.size());
    BatchSetup s;
    s.reqs = reqs;
    s.cu_extend_lens.assign(batch_size + 1, 0);
    s.cu_prefix_lens.assign(batch_size + 1, 0);
    int max_seq_len = 0;
    int c128_max_seq_len_clamp1 = 1;
    int max_extend_len = 0;
    for (int r = 0; r < batch_size; ++r) {
      const int sl = reqs[r].prefix_len + reqs[r].extend_len;
      s.cu_extend_lens[r + 1] = s.cu_extend_lens[r] + reqs[r].extend_len;
      s.cu_prefix_lens[r + 1] = s.cu_prefix_lens[r] + reqs[r].prefix_len;
      max_extend_len = std::max(max_extend_len, reqs[r].extend_len);
      max_seq_len = std::max(max_seq_len, sl);
      const int c128_seq_len = sl / kC128Ratio;
      if (c128_seq_len > c128_max_seq_len_clamp1) c128_max_seq_len_clamp1 = c128_seq_len;
    }
    s.num_tokens = s.cu_extend_lens[batch_size];
    s.max_blocks_per_seq = (max_seq_len + block_token_num - 1) / block_token_num;
    s.aligned_c128_max = ((c128_max_seq_len_clamp1 + kTopkBlockSize - 1) / kTopkBlockSize) * kTopkBlockSize;
    s.max_extend_len = max_extend_len;

    // Synthesise a c128 block_table from each req's SWA blocks (any int values are fine; -1 padding for unused tail).
    std::mt19937 gen(0xC0FFEE);
    std::uniform_int_distribution<int> blk_dist(0, 4096);
    s.block_table.assign(batch_size * s.max_blocks_per_seq, -1);
    for (int r = 0; r < batch_size; ++r) {
      const int seq_len = reqs[r].prefix_len + reqs[r].extend_len;
      const int used = (seq_len + block_token_num - 1) / block_token_num;
      for (int b = 0; b < used; ++b) {
        s.block_table[r * s.max_blocks_per_seq + b] = blk_dist(gen);
      }
    }
    return s;
  }

  static ReqDescr MakeReq(int prefix_len, int extend_len, int block_token_num, std::mt19937& gen) {
    ReqDescr r;
    r.prefix_len = prefix_len;
    r.extend_len = extend_len;
    const int total = prefix_len + extend_len;
    const int num_blocks = (total + block_token_num - 1) / block_token_num;
    std::uniform_int_distribution<int> blk_dist(0, 4096);
    for (int b = 0; b < num_blocks; ++b) {
      r.swa_block_ids.push_back(blk_dist(gen));
      r.swa_block_idx_offsets.push_back(static_cast<size_t>(b));
    }
    return r;
  }

  // Run the fused kernel and verify both c128 (block_indices + topk_lengths) and SWA outputs.
  void RunFusedCase(const std::vector<ReqDescr>& reqs, int block_token_num, bool fill_topk_with_stride) {
    const int batch_size = static_cast<int>(reqs.size());
    auto setup = BuildBatchSetup(reqs, block_token_num);
    constexpr int kTailStride = kSwaSize - 1;

    // Build raw_out_loc (per-extend-token SWA phys pos).
    std::vector<int32_t> raw_out_loc_host;
    for (int r = 0; r < batch_size; ++r) {
      for (int local = 0; local < reqs[r].extend_len; ++local) {
        const int i = reqs[r].prefix_len + local;
        raw_out_loc_host.push_back(
            HostSwaPhysPos(reqs[r].swa_block_ids, reqs[r].swa_block_idx_offsets, i, block_token_num));
      }
    }

    // Build swa_prefix_tail_block_indices buffer with shape [batch_size, kSwaSize] (last column unused).
    std::vector<int32_t> swa_prefix_tail_block_indices_host(static_cast<size_t>(batch_size) * kSwaSize, -1);
    for (int r = 0; r < batch_size; ++r) {
      const int prefix_len = reqs[r].prefix_len;
      const int tail_first_j = prefix_len - kTailStride;
      int32_t* row = swa_prefix_tail_block_indices_host.data() + static_cast<size_t>(r) * kSwaSize;
      for (int m = 0; m < kTailStride; ++m) {
        row[m] =
            HostSwaPhysPos(reqs[r].swa_block_ids, reqs[r].swa_block_idx_offsets, tail_first_j + m, block_token_num);
      }
    }

    // Reference outputs.
    std::vector<int32_t> ref_c128_block_idx(static_cast<size_t>(setup.num_tokens) * setup.aligned_c128_max, 0);
    std::vector<int32_t> ref_c128_topk_len(batch_size, 0);
    HostRefC128(setup.cu_extend_lens, setup.cu_prefix_lens, setup.block_table, batch_size, block_token_num,
                setup.max_blocks_per_seq, setup.aligned_c128_max, fill_topk_with_stride, ref_c128_block_idx,
                ref_c128_topk_len);

    std::vector<int32_t> ref_swa_block_indices(static_cast<size_t>(setup.num_tokens) * kSwaSize, 0);
    for (int r = 0; r < batch_size; ++r) {
      const int extend_begin = setup.cu_extend_lens[r];
      const int prefix_len = reqs[r].prefix_len;
      const int extend_len = reqs[r].extend_len;
      for (int local = 0; local < extend_len; ++local) {
        const int i = prefix_len + local;
        const int global_t = extend_begin + local;
        for (int k = 0; k < kSwaSize; ++k) {
          const int j = i - kSwaSize + 1 + k;
          ref_swa_block_indices[static_cast<size_t>(global_t) * kSwaSize + k] =
              HostSwaPhysPos(reqs[r].swa_block_ids, reqs[r].swa_block_idx_offsets, j, block_token_num);
        }
      }
    }

    // Allocate device buffers.
    BufferMeta d_cu_extend_lens = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size + 1)});
    BufferMeta d_cu_prefix_lens = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size + 1)});
    BufferMeta d_block_table =
        CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(setup.block_table.size())});
    BufferMeta d_raw_out_loc =
        CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {std::max<size_t>(1, raw_out_loc_host.size())});
    BufferMeta d_swa_prefix_tail_block_indices =
        CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {std::max<size_t>(1, swa_prefix_tail_block_indices_host.size())});
    BufferMeta d_c128_block_idx =
        CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(setup.num_tokens) * setup.aligned_c128_max});
    BufferMeta d_c128_topk_len = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size)});
    BufferMeta d_swa_block_idx = CreateBuffer<int32_t>(
        MemoryType::MEMORY_GPU, {std::max<size_t>(1, static_cast<size_t>(setup.num_tokens) * kSwaSize)});

    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_cu_extend_lens.data_ptr, setup.cu_extend_lens.data(),
                                       setup.cu_extend_lens.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_cu_prefix_lens.data_ptr, setup.cu_prefix_lens.data(),
                                       setup.cu_prefix_lens.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_block_table.data_ptr, setup.block_table.data(),
                                       setup.block_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    if (!raw_out_loc_host.empty()) {
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_raw_out_loc.data_ptr, raw_out_loc_host.data(),
                                         raw_out_loc_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_swa_prefix_tail_block_indices.data_ptr, swa_prefix_tail_block_indices_host.data(),
                   swa_prefix_tail_block_indices_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    // Pre-fill outputs with sentinel.
    std::vector<int32_t> sentinel_topk(batch_size, 0x7BAD7BAD);
    std::vector<int32_t> sentinel_c128(static_cast<size_t>(setup.num_tokens) * setup.aligned_c128_max, 0x7BAD7BAD);
    std::vector<int32_t> sentinel_swa(static_cast<size_t>(setup.num_tokens) * kSwaSize, 0x7BAD7BAD);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_c128_block_idx.data_ptr, sentinel_c128.data(),
                                       sentinel_c128.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_c128_topk_len.data_ptr, sentinel_topk.data(), batch_size * sizeof(int32_t),
                                       cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_swa_block_idx.data_ptr, sentinel_swa.data(),
                                       sentinel_swa.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    InvokePrepareDsv4MqaMeta(
        reinterpret_cast<int32_t*>(d_cu_extend_lens.data_ptr), reinterpret_cast<int32_t*>(d_cu_prefix_lens.data_ptr),
        reinterpret_cast<int32_t*>(d_block_table.data_ptr), reinterpret_cast<int32_t*>(d_raw_out_loc.data_ptr),
        reinterpret_cast<int32_t*>(d_swa_prefix_tail_block_indices.data_ptr), batch_size, block_token_num,
        setup.max_blocks_per_seq, setup.max_extend_len, reinterpret_cast<int32_t*>(d_swa_block_idx.data_ptr),
        reinterpret_cast<int32_t*>(d_c128_block_idx.data_ptr), reinterpret_cast<int32_t*>(d_c128_topk_len.data_ptr),
        kSwaSize, setup.aligned_c128_max, fill_topk_with_stride, /*noncausal=*/false, stream);

    std::vector<int32_t> got_c128(static_cast<size_t>(setup.num_tokens) * setup.aligned_c128_max, 0);
    std::vector<int32_t> got_topk(batch_size, 0);
    std::vector<int32_t> got_swa(static_cast<size_t>(setup.num_tokens) * kSwaSize, 0);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(got_c128.data(), d_c128_block_idx.data_ptr, got_c128.size() * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(got_topk.data(), d_c128_topk_len.data_ptr, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(got_swa.data(), d_swa_block_idx.data_ptr, got_swa.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    for (int r = 0; r < batch_size; ++r) {
      EXPECT_EQ(got_topk[r], ref_c128_topk_len[r]) << "c128_topk_len mismatch at req=" << r;
    }
    for (int t = 0; t < setup.num_tokens; ++t) {
      for (int j = 0; j < setup.aligned_c128_max; ++j) {
        EXPECT_EQ(got_c128[t * setup.aligned_c128_max + j], ref_c128_block_idx[t * setup.aligned_c128_max + j])
            << "c128_block_idx mismatch at t=" << t << " j=" << j;
      }
      for (int k = 0; k < kSwaSize; ++k) {
        EXPECT_EQ(got_swa[t * kSwaSize + k], ref_swa_block_indices[t * kSwaSize + k])
            << "swa_block_idx mismatch at t=" << t << " k=" << k;
      }
    }

    DeleteBuffer(d_cu_extend_lens);
    DeleteBuffer(d_cu_prefix_lens);
    DeleteBuffer(d_block_table);
    DeleteBuffer(d_raw_out_loc);
    DeleteBuffer(d_swa_prefix_tail_block_indices);
    DeleteBuffer(d_c128_block_idx);
    DeleteBuffer(d_c128_topk_len);
    DeleteBuffer(d_swa_block_idx);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Correctness tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(NvidiaPrepareDsv4MqaMetaTestSuit, MixedPrefillAndDecode) {
  std::mt19937 gen(0xC0FFEE);
  const int block_token_num = 128;
  std::vector<ReqDescr> reqs = {
      MakeReq(0, 200, block_token_num, gen),   // pure prefill, prefix=0
      MakeReq(64, 130, block_token_num, gen),  // partial prefill, tail < swa
      MakeReq(256, 1, block_token_num, gen),   // decode, prefix > swa
      MakeReq(512, 1, block_token_num, gen),   // decode, prefix >> swa
      MakeReq(127, 1, block_token_num, gen),   // boundary, tail = swa-1
      MakeReq(1, 1, block_token_num, gen),     // tiny prefix
  };
  RunFusedCase(reqs, block_token_num, /*fill_topk_with_stride=*/false);
}

TEST_F(NvidiaPrepareDsv4MqaMetaTestSuit, MixedPrefillAndDecode_FillTopkWithStride) {
  std::mt19937 gen(0xC0FFEE);
  const int block_token_num = 128;
  std::vector<ReqDescr> reqs = {
      MakeReq(0, 200, block_token_num, gen),
      MakeReq(64, 130, block_token_num, gen),
      MakeReq(256, 1, block_token_num, gen),
      MakeReq(512, 1, block_token_num, gen),
  };
  RunFusedCase(reqs, block_token_num, /*fill_topk_with_stride=*/true);
}

TEST_F(NvidiaPrepareDsv4MqaMetaTestSuit, LongPrefillSingleReq) {
  std::mt19937 gen(0xCAFEBABE);
  const int block_token_num = 128;
  std::vector<ReqDescr> reqs = {MakeReq(0, 4096, block_token_num, gen)};
  RunFusedCase(reqs, block_token_num, /*fill_topk_with_stride=*/false);
}

TEST_F(NvidiaPrepareDsv4MqaMetaTestSuit, AllDecode) {
  std::mt19937 gen(0xDEADBEEF);
  const int block_token_num = 128;
  std::vector<ReqDescr> reqs;
  for (int r = 0; r < 8; ++r) {
    reqs.push_back(MakeReq(1024 + r * 64, 1, block_token_num, gen));
  }
  RunFusedCase(reqs, block_token_num, /*fill_topk_with_stride=*/false);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
