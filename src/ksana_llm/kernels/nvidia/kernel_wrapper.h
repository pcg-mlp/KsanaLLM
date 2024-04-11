/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <optional>
#include <vector>

#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"

#include "ksana_llm/utils/nvidia/nccl_utils.h"

namespace ksana_llm {

// Invoke the lookup embedding.
template <typename T>
void LookupEmbedding(const void* ids, const void* offset, const void* emb, const void* pos, void* output,
                     int vocab_size, int hidden_size, int bs, int step, int vocab_id, cudaStream_t stream,
                     void* workspace_ptr = nullptr);

template <typename T>
void InvokeLayerNorm(const void* input, const void* weight, const float layernorm_eps, const int m, const int n,
                     void* output, cudaStream_t stream);

template <typename T>
void InvokeMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,
                  const void* a_ptr, const void* b_ptr, void* c_ptr, cudaStream_t& stream);

template <typename T>
void InvokeAddBiasResidual(const void* input_a, const void* input_b, const void* bias, const int m, const int n,
                           void* output, cudaStream_t stream);

template <typename T>
void InvokeSiluActivation(const void* input, const void* bias, const int m, const int n, void* output,
                          cudaStream_t stream);

template <typename T>
void AssembleLastToken(const void* input, const void* offset, const int batch_size, const int hidden_units_num,
                       void* output, cudaStream_t& stream);

template <typename T>
void AttenVarlen(void* qkv_ptr, void* rotary_embedding_pos, void* out, void* seqlen,
                 llm_kernels::nvidia::RotaryEmbeddingCuda<T>& rotary_embedding_cuda, int total_tokens, int max_tokens,
                 int batch, int num_heads, int head_size, int stride_size, int tensor_para_size, bool is_causal,
                 int rank, int block_size, void** k_list, void** v_list, void* block_offset,
                 const std::optional<void*>& alibi_slopes, cudaStream_t stream);

template <typename T>
void run_paged_attention(void* out,                // [num_seqs, num_heads, head_size]
                         void* query,              // [num_seqs, num_heads, head_size]
                         void** key_cache_ptrs,    // num_seqs,[seq_blocks]
                         void** value_cache_ptrs,  // num_seqs,[seq_blocks]
                         void* context_lens_ptr,   // [num_seqs]
                         int max_context_len, cudaStream_t stream,
                         void* cache_offsets_ptr,  // num_seqs
                         int num_seqs, int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size,
                         int batch, void* rotary_embedding_pos, int total_tokens,
                         llm_kernels::nvidia::RotaryEmbeddingCuda<T>& rotary_embedding_cuda, void* workspace,
                         size_t work_size, int rank, const std::optional<void*>& alibi_slopes, void* qkv_workspace);

template <typename T>
void CustomAllReduceInit(void** ptr, void* input, void** metas, void* rank_data, void** data_handles,
                         void** input_handles, int data_size, size_t rank_data_sz, int tp_size, int rank,
                         cudaStream_t& stream);

template <typename T>
void CustomAllReduceRun(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream);

template <typename T>
ncclDataType_t GetNcclDataType();

template <typename T>
void DataToFloat(const void* input, const int data_size, void* output, cudaStream_t& stream);

template <typename T>
void InvokePermute(void* input, void* output, std::vector<size_t> input_shape, std::vector<size_t> permutation,
                   cudaStream_t& stream);

// c = Mul(a, b)
void Mul(float* a, float* b, float* c, int n, int device_rank);

}  // namespace ksana_llm
