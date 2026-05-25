/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/store.cuh
 *
 * Copyright 2023-2024 SGLang Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {

// -----------------------------------------------------------------------------
// fused_store_indexer_cache
// -----------------------------------------------------------------------------
// Pack and write input tokens into a paged indexer KV cache.
//
// Input  : [num_tokens, 128] Float
// Cache  : paged byte array; each page is kPageBytes bytes.
//          kPageBytes = 132 * page_size
//   Per page layout:
//     [0 .. 128*page_size) : FP8-E4M3 values
//     [128*page_size .. )  : float32 scale (4 bytes per slot)
// indices : [num_tokens] IndicesT
template <typename Float, typename IndicesT, uint32_t kPageBits>
void fused_store_indexer_cache(const void* input, void* cache, const void* indices, uint32_t num_tokens,
                               int64_t page_stride, cudaStream_t stream);

// -----------------------------------------------------------------------------
// fused_gather_indexer_cache
// -----------------------------------------------------------------------------
// Gather FP8 values and per-token scales from a paged indexer KV cache into
// contiguous output buffers (reverse of fused_store_indexer_cache).
//
// Output : k_fp8   [total_tokens, 128] Fp8E4M3
//          k_scale [total_tokens] Float
// Cache  : paged byte array (same layout as fused_store_indexer_cache).
//   Per page layout:
//     [0 .. 128*page_size) : FP8-E4M3 values
//     [128*page_size .. )  : float32 scale (4 bytes per slot)
// block_table    : [batch_size, max_num_blocks_per_seq] IndicesT
// seq_len_offset : [batch_size + 1] size_t, cumulative uncompressed lengths
//                  (divided by compress_ratio internally)
template <typename Float, typename IndicesT, uint32_t kPageBits>
void fused_gather_indexer_cache(void* k_fp8, void* k_scale, const void* cache, const void* block_table,
                                const size_t* seq_len_offset, uint32_t batch_size, uint32_t total_tokens,
                                int64_t page_stride, int32_t max_num_blocks_per_seq, int32_t compress_ratio,
                                int32_t gather_offset, cudaStream_t stream);

#define DISPATCH_STORE_BY_PAGE_BITS(page_bits, func, ...)              \
  switch (page_bits) {                                                 \
    case 1:                                                            \
      CUDA_CHECK_LAST_ERROR(func<SCALAR_T, int32_t, 1>(__VA_ARGS__));  \
      break;                                                           \
    case 6:                                                            \
      CUDA_CHECK_LAST_ERROR(func<SCALAR_T, int32_t, 6>(__VA_ARGS__));  \
      break;                                                           \
    case 8:                                                            \
      CUDA_CHECK_LAST_ERROR(func<SCALAR_T, int32_t, 8>(__VA_ARGS__));  \
      break;                                                           \
    default:                                                           \
      KLLM_THROW(fmt::format("Unsupported page_bits: {}", page_bits)); \
  }

}  // namespace nvidia
}  // namespace llm_kernels
