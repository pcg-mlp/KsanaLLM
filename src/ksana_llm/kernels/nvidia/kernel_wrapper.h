/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <optional>
#include <vector>

#include "csrc/kernels/nvidia/asymmetric_gemm/asymmetric_gemm_wrapper.h"
#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#include "csrc/utils/quant_type.h"

#include "ksana_llm/utils/nvidia/nccl_utils.h"

namespace ksana_llm {

template <typename T, llm_kernels::nvidia::WeightType WT>
void GetFpAIntBGroupCutlassGemmWorkspaceSize(size_t m, size_t n, size_t k, size_t& ws_bytes);

template <typename T, llm_kernels::nvidia::WeightType WT>
void InvokeFpAIntBGroupCutlassGemm(void* output, const void* input, const void* weight, const void* scales, void* ws,
                                   size_t m, size_t n, size_t k, size_t groupsize, size_t config_index,
                                   cudaStream_t stream);

template <typename T, llm_kernels::nvidia::WeightType WT>
size_t InvokeFpAIntBGroupCutlassGemmConfigProfile(size_t warmup, size_t iter, void* output, const void* input,
                                                  const void* weight, const void* scales, void* ws, size_t m, size_t n,
                                                  size_t k, size_t groupsize, cudaStream_t stream);

template <typename T, llm_kernels::nvidia::WeightType WT>
bool GetFpAIntBGroupCudaGemmSupported();

template <typename T, llm_kernels::nvidia::WeightType WT>
void InvokeFpAIntBGroupCudaGemm(void* output, const void* input, const void* weight, const void* scales, size_t m,
                                size_t n, size_t k, size_t groupsize, cudaStream_t stream);

// Invoke the lookup embedding.
template <typename T>
void LookupEmbedding(const void* input_ids, const void* ids_offsets, const void* prefix_offsets, const void* emb,
                     const void* pos, const void* steps, void* output, const T emb_scale, int vocab_size,
                     int hidden_size, int bs, int vocab_id, cudaStream_t stream, void* workspace_ptr = nullptr);

// Layernorm without bias computes rmsnorm.
template <typename T>
void InvokeLayerNorm(const void* input, const void* weight, const void* bias, const float layernorm_eps, const int m,
                     const int n, void* output, cudaStream_t stream);

template <typename T>
void InvokeMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,
                  const void* a_ptr, const void* b_ptr, void* c_ptr, cudaStream_t& stream);

template <typename T>
void InvokeAddBiasResidual(const void* input_a, const void* input_b, const void* bias, const int m, const int n,
                           void* output, cudaStream_t stream);

// Invoke activation in-place, `output` must be the same as `input`.
template <template <typename T> class Activation, typename T>
void InvokeGatedActivation(const void* input, const void* bias, const void* gated_weights, const void* gated_bias,
                           const int m, const int n, void* output, cudaStream_t stream);

template <typename T>
void AssembleLastToken(const void* inputs, const void* offsets, const void* prefix_offsets, const int batch_size,
                       const int hidden_units_num, void* output, cudaStream_t& stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void AttenVarlen(void* qkv_ptr, void* rotary_embedding_pos, void* rotary_embedding_mask, void* out, void* seqlen,
                 std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
                 int total_tokens, int max_tokens, int batch, int num_heads, int num_kv_heads, int head_size,
                 int stride_size, float k_scale, float v_scale, int tensor_para_size, bool is_causal, int rank,
                 int block_size, void** k_list, void** v_list, void* prefix_offsets, void* block_offsets,
                 const std::optional<void*>& alibi_slopes, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokePagedAttention(void* out,                // [num_seqs, num_heads, head_size]
                          void* query,              // [num_seqs, num_heads, head_size]
                          void** key_cache_ptrs,    // num_seqs,[seq_blocks]
                          void** value_cache_ptrs,  // num_seqs,[seq_blocks]
                          void* context_lens_ptr,   // [num_seqs]
                          int max_context_len, cudaStream_t stream,
                          void* cache_offsets_ptr,  // num_seqs
                          int num_seqs, int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size,
                          float k_scale, float v_scale, int batch, void* rotary_embedding_pos,
                          void* rotary_embedding_mask, int total_tokens,
                          std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
                          void* workspace, size_t work_size, int rank, const std::optional<void*>& alibi_slopes,
                          void* qkv_workspace);

template <typename T>
void CustomAllReduceInit(void** ptr, void* input, void** metas, void* rank_data, void** data_handles,
                         void** input_handles, int data_size, size_t rank_data_sz, int tp_size, int rank,
                         cudaStream_t& stream);

template <typename T>
void CustomAllReduceRun(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream);

template <typename T>
ncclDataType_t GetNcclDataType();

template <typename T>
void DataToFloat(const void* input, const int data_size, const size_t vocab_size, const size_t vocab_size_pad,
                 void* output, cudaStream_t& stream);

template <typename T>
void InvokePermute(void* input, void* output, std::vector<size_t> input_shape, std::vector<size_t> permutation,
                   cudaStream_t& stream);

// c = Mul(a, b)
void Mul(float* a, float* b, float* c, int n, int device_rank);

void CalcLogprobs(float* logits, float* temperatures, int vocab_size, int bs, int logprobs_num, float* logprobs,
                  int64_t* token_ids);

#ifdef ENABLE_FP8
template <typename T>
void Fp8E4m3Quantize(int num_channels, int channel_size, const T* input_ptr, void* quant_ptr, float* scale_ptr,
                     bool is_static, cudaStream_t& stream);

template <typename T>
void Fp8QuantizedMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,
                        const void* a_ptr, const void* a_scale, const void* b_ptr, const void* b_scale, T* c_ptr,
                        cudaStream_t& stream, void* workspace);

void RescaleFp8E4m3(void* input, void* output, size_t n, const float* input_scale, const float* output_scale,
                    cudaStream_t& stream);
#endif

size_t InvokeGetCublasWorkspaceSize();
}  // namespace ksana_llm
