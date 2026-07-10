/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {

/**
 * @brief Compute DSv4 MQA c128 block_indices/topk_lengths AND SWA block_indices in one fused kernel launch.
 *
 * Single-launch fused kernel with grid (batch_size, y_tiles, 2):
 *   blockIdx.z == 0 -> c128 path: writes c128_block_indices[t, 0..stride) and
 *                                 c128_topk_lengths[req] (once per req).
 *   blockIdx.z == 1 -> SWA path : writes swa_block_indices[t, 0..swa_size).
 *
 * c128 path:
 *   c128_block_indices[t, j] = (j < c128_seq_len) ? (block_id << shift) | (j & mask) : -1
 *   c128_topk_lengths [req]  = fill_topk_with_stride ? c128_block_indices_stride : c128_seq_len
 *
 * SWA path (hybrid CPU + GPU):
 *   For local_idx = t - extend_begin_r and m = local_idx + k:
 *     m >= swa_size - 1 -> raw_out_loc[extend_begin_r + (m - (swa_size - 1))]
 *     m <  swa_size - 1 -> swa_prefix_tail_block_indices[req_idx * swa_size + m]
 *
 * Inputs:
 *   cu_extend_lens                 [batch_size + 1]                    cumulative extend_len (int32)
 *   cu_prefix_lens                 [batch_size + 1]                    cumulative prefix_len (int32)
 *   block_table                    [batch_size, block_table_stride]    padded with -1 (int32)
 *   raw_out_loc                    [Σ extend_len]                      per-extend-token SWA phys pos
 *   batch_size                                                         number of requests
 *   block_token_num                                                    tokens per cache block; must be a power of two
 *   max_extend_len                                                     upper bound on max_r(extend_len_r)
 *   swa_size                                                           must equal kSwaSize (128)
 *   fill_topk_with_stride                                              true  (flash)
 *                                                                      false (paged)
 *
 * Outputs:
 *   swa_block_indices   [Σ extend_len, swa_size]                    int32, row-major
 *   c128_block_indices  [Σ extend_len, c128_block_indices_stride]   int32, row-major, padded with -1
 *   c128_topk_lengths   [batch_size]                                int32, one per request
 *
 * Caller must ensure raw_out_loc and swa_prefix_tail_block_indices have been uploaded on the same stream
 * prior to invocation.
 */
void InvokePrepareDsv4MqaMeta(const int32_t* cu_extend_lens, const int32_t* cu_prefix_lens, const int32_t* block_table,
                              const int32_t* raw_out_loc, const int32_t* swa_prefix_tail_block_indices, int batch_size,
                              int block_token_num, int block_table_stride, int max_extend_len,
                              int32_t* swa_block_indices, int32_t* c128_block_indices, int32_t* c128_topk_lengths,
                              int swa_size, int c128_block_indices_stride, bool fill_topk_with_stride, bool noncausal,
                              cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
