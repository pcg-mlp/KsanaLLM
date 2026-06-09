/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/attention_meta/prepare_dsv4_mqa_meta.h"

#include <cassert>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

namespace {

constexpr int kC128Ratio = 128;
constexpr int kC128RatioBits = 7;

// Fused kernel grid: (batch_size, y_tiles, 2)
//   blockIdx.z == 0: c128 path  (writes c128_block_indices + c128_topk_lengths)
//   blockIdx.z == 1: SWA path   (writes swa_block_indices)
//
// Block: (kThreadsX, kTokensPerBlock) = (128, 4) — sized to hold the SWA path's full row of
// `swa_size` columns under one thread per column. The c128 path reuses these threads and
// strides along its (typically larger) row width with a step of kThreadsX.
constexpr int kThreadsX = 128;
constexpr int kTokensPerBlock = 4;
constexpr int kThreadsTotal = kThreadsX * kTokensPerBlock;

// ── c128 path body (called when blockIdx.z == 0) ─────────────────────────────
__device__ inline void RunC128Path(const int32_t* __restrict__ cu_extend_lens,
                                   const int32_t* __restrict__ cu_prefix_lens, const int32_t* __restrict__ block_table,
                                   const int block_token_num_bits, const int block_table_stride,
                                   int32_t* __restrict__ c128_block_indices, int32_t* __restrict__ c128_topk_lengths,
                                   const int c128_block_indices_stride, const bool fill_topk_with_stride,
                                   int32_t* row_template) {
  static_assert(kC128Ratio == 128, "kC128Ratio must be 128");

  const int req_idx = blockIdx.x;
  const int y_tile = blockIdx.y;

  const int extend_begin = cu_extend_lens[req_idx];
  const int extend_len = cu_extend_lens[req_idx + 1] - extend_begin;
  const int prefix_len = cu_prefix_lens[req_idx + 1] - cu_prefix_lens[req_idx];

  const int local_idx_base = y_tile * kTokensPerBlock;
  if (local_idx_base >= extend_len) {
    return;
  }

  const int c128_seq_len = (prefix_len + extend_len) >> kC128RatioBits;

  // Per-request scalar output: written once by the (y_tile == 0) block's first thread.
  if (y_tile == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    c128_topk_lengths[req_idx] = fill_topk_with_stride ? c128_block_indices_stride : c128_seq_len;
  }

  const int32_t* req_block_table = block_table + req_idx * block_table_stride;
  const int c128_slots_per_block_bits = block_token_num_bits - kC128RatioBits;
  const int c128_slots_per_block_mask = (1 << c128_slots_per_block_bits) - 1;

  // Build per-request row template in smem: row_template[j] = (j < c128_seq_len) ? phys_pos(j) : -1.
  const int tid_flat = threadIdx.y * blockDim.x + threadIdx.x;
  for (int j = tid_flat; j < c128_block_indices_stride; j += kThreadsTotal) {
    int32_t v = -1;
    if (j < c128_seq_len) {
      const int j_token = ((j + 1) << kC128RatioBits) - 1;
      const int j_block = req_block_table[j_token >> block_token_num_bits];
      v = (j_block << c128_slots_per_block_bits) | (j & c128_slots_per_block_mask);
    }
    row_template[j] = v;
  }
  __syncthreads();

  // Each (token, j) tile copies template -> output, masking columns past this token's c128_len.
  const int local_idx = local_idx_base + threadIdx.y;
  if (local_idx >= extend_len) {
    return;
  }
  const int i = prefix_len + local_idx;
  const int global_i = extend_begin + local_idx;
  const int c128_len = (i + 1) >> kC128RatioBits;

  int32_t* row = c128_block_indices + static_cast<size_t>(global_i) * c128_block_indices_stride;
  for (int j = threadIdx.x; j < c128_block_indices_stride; j += kThreadsX) {
    row[j] = (j < c128_len) ? row_template[j] : -1;
  }
}

// ── SWA path body (called when blockIdx.z == 1) ──────────────────────────────
__device__ inline void RunSwaPath(const int32_t* __restrict__ cu_extend_lens, const int32_t* __restrict__ raw_out_loc,
                                  const int32_t* __restrict__ swa_prefix_tail_block_indices,
                                  int32_t* __restrict__ swa_block_indices, const int swa_size) {
  const int req_idx = blockIdx.x;
  const int y_tile = blockIdx.y;

  const int extend_begin = cu_extend_lens[req_idx];
  const int extend_len = cu_extend_lens[req_idx + 1] - extend_begin;
  const int local_idx = y_tile * kTokensPerBlock + threadIdx.y;
  if (local_idx >= extend_len) {
    return;
  }

  // SWA path uses one thread per column; threads beyond swa_size are idle.
  const int k = threadIdx.x;
  if (k >= swa_size) {
    return;
  }

  // tail_stride = number of prefix-tail columns at the left of the window.
  // m          = position of column k in the virtual [prefix_tail | extend] row.
  // global_t   = absolute extend-token index, used to address the output row.
  const int tail_stride = swa_size - 1;
  const int m = local_idx + k;
  const int global_t = extend_begin + local_idx;

  int32_t v;
  if (m >= tail_stride) {
    // Extend region: covers prefix-included absolute positions [prefix_len, prefix_len + local_idx].
    v = raw_out_loc[extend_begin + (m - tail_stride)];
  } else {
    // Prefix-tail region: host-uploaded buffer holds prefix slots
    // [prefix_len - tail_stride, prefix_len) packed at indices [0, tail_stride).
    v = swa_prefix_tail_block_indices[req_idx * swa_size + m];
  }
  swa_block_indices[static_cast<size_t>(global_t) * swa_size + k] = v;
}

// ── Fused kernel entry point ─────────────────────────────────────────────────
__global__ void PrepareDsv4MqaMetaKernel(
    const int32_t* __restrict__ cu_extend_lens, const int32_t* __restrict__ cu_prefix_lens,
    const int32_t* __restrict__ block_table, const int32_t* __restrict__ raw_out_loc,
    const int32_t* __restrict__ swa_prefix_tail_block_indices, const int block_token_num_bits,
    const int block_table_stride, int32_t* __restrict__ swa_block_indices, int32_t* __restrict__ c128_block_indices,
    int32_t* __restrict__ c128_topk_lengths, const int swa_size, const int c128_block_indices_stride,
    const bool fill_topk_with_stride) {
  extern __shared__ int32_t row_template[];

  if (blockIdx.z == 0) {
    RunC128Path(cu_extend_lens, cu_prefix_lens, block_table, block_token_num_bits, block_table_stride,
                c128_block_indices, c128_topk_lengths, c128_block_indices_stride, fill_topk_with_stride, row_template);
  } else {
    RunSwaPath(cu_extend_lens, raw_out_loc, swa_prefix_tail_block_indices, swa_block_indices, swa_size);
  }
}

// Returns log2(x) for power-of-two x; -1 otherwise.
int Log2IfPow2(int x) {
  if (x <= 0 || (x & (x - 1)) != 0) {
    return -1;
  }
  int b = 0;
  while ((1 << b) < x) {
    ++b;
  }
  return b;
}

}  // namespace

void InvokePrepareDsv4MqaMeta(const int32_t* cu_extend_lens, const int32_t* cu_prefix_lens, const int32_t* block_table,
                              const int32_t* raw_out_loc, const int32_t* swa_prefix_tail_block_indices, int batch_size,
                              int block_token_num, int block_table_stride, int max_extend_len,
                              int32_t* swa_block_indices, int32_t* c128_block_indices, int32_t* c128_topk_lengths,
                              int swa_size, int c128_block_indices_stride, bool fill_topk_with_stride,
                              cudaStream_t stream) {
  if (batch_size <= 0 || max_extend_len <= 0) {
    return;
  }

  const int block_token_num_bits = Log2IfPow2(block_token_num);
  assert(block_token_num_bits >= kC128RatioBits && "block_token_num must be a power of two and >= kC128Ratio (128)");
  assert((c128_block_indices_stride & 3) == 0 && "c128_block_indices_stride must be a multiple of 4");
  assert(swa_size > 0 && swa_size <= kThreadsX && "swa_size must be <= kThreadsX (128)");

  const int y_tiles = (max_extend_len + kTokensPerBlock - 1) / kTokensPerBlock;
  dim3 grid(batch_size, y_tiles, 2);
  dim3 block(kThreadsX, kTokensPerBlock);
  const size_t smem_bytes = static_cast<size_t>(c128_block_indices_stride) * sizeof(int32_t);
  PrepareDsv4MqaMetaKernel<<<grid, block, smem_bytes, stream>>>(
      cu_extend_lens, cu_prefix_lens, block_table, raw_out_loc, swa_prefix_tail_block_indices, block_token_num_bits,
      block_table_stride, swa_block_indices, c128_block_indices, c128_topk_lengths, swa_size, c128_block_indices_stride,
      fill_topk_with_stride);
}

}  // namespace nvidia
}  // namespace llm_kernels
