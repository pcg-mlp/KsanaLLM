/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {


// @param block_ids_array     [req_num] device pointer-to-device-pointer;
//                            entry r points to the request-r KV block-id array
// @param prefix_lens         [req_num] int32 device: cached token count / req
// @param seq_lens            [req_num] int32 device: total token count / req
// @param out                 flat int32 device buffer; layout is requests
//                            laid out back-to-back according to `out_offsets`
// @param out_offsets         [req_num] int32 device; out_offsets[r] is the
//                            row index at which request r's slab starts
//                            (i.e. prefix sum of extend_len_0..extend_len_{r-1})
// @param req_num             number of requests (0 is a valid no-op)
// @param sliding_window_size runtime sliding-window size (a.k.a. kSwa); the
//                            number of output columns per row is
//                            `sliding_window_size`. Must be >= 2.
// @param block_token_num     tokens per KV block (shared across the batch)
// @param stream              CUDA stream
void InvokeSwaTopkIdxs(const int* const* block_ids_array, const int* prefix_lens, const int* seq_lens, int* out,
                            const int* out_offsets, int req_num, int sliding_window_size, int block_token_num,
                            cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
