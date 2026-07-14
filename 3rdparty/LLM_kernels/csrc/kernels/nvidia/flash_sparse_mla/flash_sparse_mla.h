/*
 * Adapted from
 * https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/pybind.cpp
 *
 * Copyright (c) 2025 DeepSeek
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/utils/quant_type.h"

namespace llm_kernels {
namespace nvidia {

int GetNumSmParts(int num_heads_q, int seqlen_q);

int GetDecodingSchedMetaSize();

// Wrapper of run_get_decoding_sched_meta_kernel, run_flash_splitkv_mla_fp8_sparse_kernel and
// flash_fwd_mla_combine_kernel
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeFlashSparseMlaWithKVCache(
    void* q_ptr,                       // batch_size x seqlen_q x num_heads_q x head_size_k
    CACHE_T* kcache_ptr,               // num_blocks x block_size x num_heads_k x bytes_per_token
    int* indices_ptr,                  // batch_size x seqlen_q x topk
    int* tile_scheduler_metadata_ptr,  // num_sm_parts x sizeof(DecodingSchedMeta)/sizeof(int)
    int* num_splits_ptr,               // batch_size + 1
    bool sched_meta_initialized,       // initialize sched_meta only on the first call
    int batch_size, int seqlen_q, int num_heads_q, int head_size_k, int num_blocks, int block_size, int num_heads_k,
    int kcache_batch_stride,  // stride of kcache_ptr in dimension 0
    int topk, int head_size_v, float softmax_scale, cudaStream_t stream, float* workspace_ptr, SCALAR_T* out_ptr,
    int* topk_length_ptr = nullptr,  // batch_size x seqlen_q (optioanl)
    float* attn_sink_ptr = nullptr,  // num_heads_q (optioanl)
    CACHE_T* extra_kcache_ptr =
        nullptr,                       // extra_num_blocks x extra_block_size x num_heads_k x bytes_per_token (optional)
    int* extra_indices_ptr = nullptr,  // batch_size x seqlen_q x extra_topk (optioanl)
    int* extra_topk_length_ptr = nullptr,  // batch_size (optioanl)
    int extra_num_blocks = 0, int extra_block_size = 0, int extra_topk = 0);

}  // namespace nvidia
}  // namespace llm_kernels
