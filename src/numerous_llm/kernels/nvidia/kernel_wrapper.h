/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <optional>
#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/paged_attention/paged_attention.h"

namespace numerous_llm {

void LookupEmbedding(const void* ids, const void* offset, const void* emb, const void* pos, void* output,
                     int vocab_size, int hidden_size, int bs, int step, int vocab_id, cudaStream_t stream);

void InvokeLayerNorm(const void* input, const void* weight, const float layernorm_eps, const int m, const int n,
                     void* output, cudaStream_t stream);

void InvokeMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,
                  const void* a_ptr, const void* b_ptr, void* c_ptr, cudaStream_t& stream);

void InvokeAddBiasResidual(const void* input, const void* bias, const int m, const int n, void* output,
                           cudaStream_t stream);

void InvokeSiluActivation(const void* input, const void* bias, const int m, const int n, void* output,
                          cudaStream_t stream);

void AssembleLastToken(const void* input, const void* offset, const int batch_size, const int hidden_units_num,
                       void* output, cudaStream_t& stream);

void AttenVarlen(void* q, void* k, void* v, void* out, void* seqlen, int total_tokens, int max_tokens, int batch,
                 int num_heads, int head_size, bool is_causal, int rank, cudaStream_t stream);

template <typename T>
void run_paged_attention(void* out,                     // [num_seqs, num_heads, head_size]
                         const void* query,             // [num_seqs, num_heads, head_size]
                         void** key_cache_ptrs,         // num_seqs,[seq_blocks]
                         void** value_cache_ptrs,       // num_seqs,[seq_blocks]
                         const void* context_lens_ptr,  // [num_seqs]
                         int max_context_len, cudaStream_t stream,
                         const void* cache_offsets_ptr,  // num_seqs
                         int num_seqs, int num_heads, int head_size, int num_kv_heads, int block_size, void* workspace,
                         size_t work_size, const std::optional<void*>& alibi_slopes) {
  const float* alibi_slopes_ptr =
      reinterpret_cast<const float*>(alibi_slopes.has_value() ? alibi_slopes.value() : nullptr);

  llm_kernels::nvidia::PagedAttentionCuda<T> op;
  op.SetConfig(num_kv_heads, num_heads, head_size, block_size);

  op.SetInput(reinterpret_cast<T*>(out), reinterpret_cast<const T*>(query), reinterpret_cast<T**>(key_cache_ptrs),
              reinterpret_cast<T**>(value_cache_ptrs), reinterpret_cast<const int*>(cache_offsets_ptr),
              reinterpret_cast<const int*>(context_lens_ptr), max_context_len, num_seqs, stream, workspace, work_size,
              alibi_slopes_ptr);
  op.Forward();
}

}  // namespace numerous_llm
