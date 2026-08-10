/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * hpc-ops paged prefill attention implementation (reads K/V from the paged KV cache).
 * Source: https://github.com/Tencent/hpc-ops (submodule pinned to 8d300e1)
 * ==============================================================================*/

#include "csrc/kernels/nvidia/attention/hpc_attention/hpc_attention.h"

#include <cuda_runtime_api.h>

#include "src/attention/prefill/prefill.h"

namespace llm_kernels {
namespace nvidia {

namespace {

// Scratch for the batched TMA descriptors updated per launch. Only the per-batch q/y descriptors
// (2 * TmaDescriptor, 128 bytes each) are staged through this buffer; k/v descriptors are passed by
// value. 4 * batch * 64 == batch * 2 * 128 bytes, matching the contiguous prefill path.
struct PagedPrefillScratch {
  void* tmas = nullptr;
  size_t tmas_bytes = 0;

  ~PagedPrefillScratch() {
    if (tmas) cudaFree(tmas);
  }

  void EnsureTmas(size_t need) {
    if (need > tmas_bytes) {
      if (tmas) cudaFree(tmas);
      cudaMalloc(&tmas, need);
      tmas_bytes = need;
    }
  }
};

thread_local PagedPrefillScratch g_paged_prefill_scratch;

}  // namespace

void HpcPagedPrefillAttention(void* out, const void* q, void* kcache, void* vcache, const int* block_table,
                              const int* seqlens_kvcache, const int* cu_seqlens_q, int batch, int total_seq_q,
                              int max_seq_q, int num_heads, int num_kv_heads, int head_size, int block_size,
                              int num_kvcache_blocks, int max_blocks_per_seq, int ldq, cudaStream_t stream) {
  if (batch <= 0 || total_seq_q <= 0) return;

  g_paged_prefill_scratch.EnsureTmas(4 * batch * 64);

  const int ldY = num_heads * head_size;  // output is packed [total_seq_q, num_heads, head_size]
  const int ldQ = ldq;                     // q is a view into the fused QKV buffer (caller-supplied stride)
  const int ldK = block_size * num_kv_heads * head_size;
  const int ldV = block_size * num_kv_heads * head_size;

  hpc::attention::attention_with_kvcache_prefill_bf16_async(
      out, q, kcache, vcache, cu_seqlens_q, block_table, seqlens_kvcache, g_paged_prefill_scratch.tmas, batch,
      total_seq_q, max_seq_q, /*num_dim_qk=*/head_size, /*num_dim_v=*/head_size, /*num_head_q=*/num_heads,
      /*num_head_kv=*/num_kv_heads, num_kvcache_blocks, block_size, /*num_seq_max_blocks=*/max_blocks_per_seq, ldY, ldQ,
      ldK, ldV, stream);
}

}  // namespace nvidia
}  // namespace llm_kernels
