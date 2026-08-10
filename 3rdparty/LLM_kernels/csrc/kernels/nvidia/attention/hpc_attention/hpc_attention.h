/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * hpc-ops attention kernel interface (decode + prefill).
 * Source: https://github.com/Tencent/hpc-ops (submodule pinned to 8d300e1)
 * SM90+ (Hopper) only. Compiled as static lib with isolated cutlass 4.4.2.
 * ==============================================================================*/
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace llm_kernels {
namespace nvidia {

// Paged decode: q attends to the full cached K/V for each sequence, read via block_table.
// KV cache layout: [num_blocks, block_size, num_kv_heads, head_size]. bf16 activations, non-fp8
// KV cache only, head_size=128, block_size in {32, 64}, GQA group (num_heads/num_kv_heads) in
// {4, 8}, decode q_seq_len typically 1-3. ldq is the row stride of q in the (fused) query buffer:
// for KsanaLLM's fused QKV this is (num_heads + 2 * num_kv_heads) * head_size.
void HpcPagedDecodeAttention(void* out, const void* q, void* kcache, void* vcache, const int* block_table,
                             const int* context_lens, int batch, int q_seq_len, int num_heads, int num_kv_heads,
                             int head_size, int block_size, int max_blocks_per_seq, int num_kvcache_blocks, int ldq,
                             cudaStream_t stream);

// Paged (blocked) contiguous prefill reading K/V from the paged KV cache via block_table.
// Supports chunked / prefix-cache prefill: cu_seqlens_q are the cumulative query offsets of the
// current chunk ({batch+1}, int32), seqlens_kvcache are the per-sequence TOTAL context lengths in
// cache (prefix + current chunk, {batch}, int32), and block_table maps each sequence to its cache
// blocks ({batch, max_blocks_per_seq}, int32). The causal mask is bottom-right aligned so the
// current chunk's queries attend to the whole cached context. The current chunk's K/V must already
// be written into the paged cache before calling. KV cache layout: [num_blocks, block_size,
// num_kv_heads, head_size]. bf16 only, head_size=128, block_size in {32, 64}.
// ldq is the row stride of q in the (fused) query buffer: for KsanaLLM's fused QKV this is
// (num_heads + 2 * num_kv_heads) * head_size, not num_heads * head_size.
void HpcPagedPrefillAttention(void* out, const void* q, void* kcache, void* vcache, const int* block_table,
                              const int* seqlens_kvcache, const int* cu_seqlens_q, int batch, int total_seq_q,
                              int max_seq_q, int num_heads, int num_kv_heads, int head_size, int block_size,
                              int num_kvcache_blocks, int max_blocks_per_seq, int ldq, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
