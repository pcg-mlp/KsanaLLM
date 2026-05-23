/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/swa_topk_idxs/swa_topk_idxs.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {

static constexpr int kBlockSize = 128;

// 每个block处理一个请求， 每个thread并发处理同一个token的sliding_window信息.
__launch_bounds__(kBlockSize, 8) __global__
    void SwaTopkIdxsKernel(const int* const* __restrict__ block_ids_array, const int* __restrict__ prefix_lens,
                                const int* __restrict__ seq_lens, int* __restrict__ out,
                                const int* __restrict__ out_offsets, const int block_token_num,
                                const int sliding_window_size) {
  __shared__ const int* s_block_ids;
  __shared__ int s_prefix_len;
  __shared__ int s_seq_len;
  __shared__ int s_out_row_base;

  const int req_id = blockIdx.x;

  if (threadIdx.x == 0) {
    s_block_ids = block_ids_array[req_id];
    s_prefix_len = __ldg(prefix_lens + req_id);
    s_seq_len = __ldg(seq_lens + req_id);
    s_out_row_base = __ldg(out_offsets + req_id);
  }
  __syncthreads();

  const int prefix_len = s_prefix_len;
  const int seq_len = s_seq_len;
  const int extend_len = seq_len - prefix_len;
  if (extend_len <= 0) {
    return;
  }

  const int* const __restrict__ block_ids = s_block_ids;
  int* const __restrict__ out_r = out + static_cast<int64_t>(s_out_row_base) * sliding_window_size;

  const int64_t total_work = static_cast<int64_t>(extend_len) * static_cast<int64_t>(sliding_window_size);

  int64_t idx = threadIdx.x;

#pragma unroll 1
  for (; idx < total_work; idx += kBlockSize) {
    // row: token index.
    const int row = static_cast<int>(idx / sliding_window_size);
    // sliding_window index
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * sliding_window_size);

    const int i = prefix_len + row;                   // token index
    const int j = i - sliding_window_size + 1 + col;  // sliding_window index

    int v;
    if (j < 0) {
      v = -1;
    } else {
      const int block_slot = j / block_token_num;
      const int in_block = j - block_slot * block_token_num;
      v = __ldg(block_ids + block_slot) * block_token_num + in_block;
    }
    out_r[idx] = v;
  }
}

// -----------------------------------------------------------------------------
// Host launcher — BATCHED, runtime sliding_window_size.
// out_offsets: req 的extend_len的累积和
// -----------------------------------------------------------------------------
void InvokeSwaTopkIdxs(const int* const* block_ids_array, const int* prefix_lens, const int* seq_lens, int* out,
                            const int* out_offsets, int req_num, int sliding_window_size, int block_token_num,
                            cudaStream_t stream) {
  // Early-exit at the batch level.
  if (req_num <= 0 || block_token_num <= 0 || sliding_window_size <= 1) {
    return;
  }

  const dim3 grid(static_cast<unsigned int>(req_num));
  const dim3 block(kBlockSize);

  SwaTopkIdxsKernel<<<grid, block, 0, stream>>>(block_ids_array, prefix_lens, seq_lens, out, out_offsets,
                                                     block_token_num, sliding_window_size);
}

}  // namespace nvidia
}  // namespace llm_kernels
