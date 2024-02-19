/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

#include <fstream>
#include <iostream>

#include "flash_api.h"

#include "csrc/kernels/nvidia/activation/activation.h"
#include "csrc/kernels/nvidia/add/add.h"
#include "csrc/kernels/nvidia/assemble_last_token/assemble_last_token.h"
#include "csrc/kernels/nvidia/cast/cast.h"
#include "csrc/kernels/nvidia/embedding/embedding.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "csrc/kernels/nvidia/layernorm/layernorm.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/paged_attention.h"

#include "ksana_llm/utils/nvidia/cuda_utils.h"

namespace ksana_llm {

void LookupEmbedding(const void* ids, const void* offset, const void* emb, const void* pos, void* output,
                     int vocab_size, int hidden_size, int bs, int step, int vocab_id, cudaStream_t stream) {
  llm_kernels::nvidia::LookupFusedEmbeddingWithCSRInputs<half>(
      reinterpret_cast<half*>(output), reinterpret_cast<const half*>(emb), reinterpret_cast<const half*>(pos), {},
      reinterpret_cast<const int32_t*>(ids), step, reinterpret_cast<const size_t*>(offset), bs, hidden_size, vocab_size,
      vocab_id, stream);
}

void InvokeLayerNorm(const void* input, const void* weight, const float layernorm_eps, const int m, const int n,
                     void* output, cudaStream_t stream) {
  half* beta = nullptr;
  llm_kernels::nvidia::InvokeLayerNorm<half>(reinterpret_cast<half*>(output), reinterpret_cast<const half*>(input),
                                             reinterpret_cast<const half*>(weight), beta, layernorm_eps, m, n, stream);
}

void InvokeMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,
                  const void* a_ptr, const void* b_ptr, void* c_ptr, cudaStream_t& stream) {
  CUDA_CHECK(llm_kernels::nvidia::InvokeCublasGemm(cublas_handle, cublaslt_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                                   b_ptr, n, CUDA_R_16F, a_ptr, k, CUDA_R_16F, c_ptr, n, CUDA_R_16F,
                                                   CUDA_R_32F, stream));
}

void InvokeAddBiasResidual(const void* input_a, const void* input_b, const int m, const int n, void* output,
                           cudaStream_t stream) {
  llm_kernels::nvidia::InvokeAddBiasResidual<half>(
      reinterpret_cast<half*>(output), reinterpret_cast<const half*>(input_a), reinterpret_cast<const half*>(input_b),
      nullptr, nullptr, nullptr, nullptr, m, n, stream);
}

void InvokeSiluActivation(const void* input, const void* gated_weights, const int m, const int n, void* output,
                          cudaStream_t stream) {
  const int* ia3_tasks = nullptr;
  const half* bias = nullptr;
  const half* ia3_weights = nullptr;
  const half* gated_bias = nullptr;
  const int int8_mode = 0;
  const int* padding_offset = nullptr;
  const int seq_len = 0;
  const float* activation_in = nullptr;
  const float* activation_out = nullptr;
  CUDA_CHECK(cudaMemcpyAsync(output, input, sizeof(half) * m * n, cudaMemcpyDeviceToDevice, stream));
  llm_kernels::nvidia::InvokeGenericActivation<llm_kernels::nvidia::SiluActivation, half, half>(
      reinterpret_cast<half*>(output), bias, reinterpret_cast<const half*>(gated_weights), gated_bias, ia3_tasks,
      ia3_weights, m, n, int8_mode, activation_in, activation_out, padding_offset, seq_len, stream);
}

void AttenVarlen(void* qkv_ptr, void* rotary_embedding_pos, void* out, void* seqlen,
                 llm_kernels::nvidia::RotaryEmbeddingCuda<half>& rotary_embedding_cuda, int total_tokens,
                 int max_tokens, int batch, int num_heads, int head_size, bool is_causal, int rank, int block_size,
                 void** k_list, void** v_list, void* block_offset, cudaStream_t stream) {
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat16);
  torch::Tensor qkv_tensor = torch::from_blob(qkv_ptr, {total_tokens, num_heads * head_size * 3}, options);
  auto tt = qkv_tensor.split(qkv_tensor.size(-1) / 3, -1);

  c10::optional<at::Tensor> out_tensor = torch::from_blob(out, {total_tokens, num_heads, head_size}, options);
  auto int_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt64);
  torch::Tensor seqlen_tensor = torch::from_blob(seqlen, {batch + 1}, int_options);

  // rotary embedding
  // TODO: 临时实现: 使用 contiguous() 将离散的 q, k 迁移到连续空间
  torch::Tensor q_tensor = tt[0].contiguous();
  torch::Tensor k_tensor = tt[1].contiguous();
  torch::Tensor v_tensor = tt[2].contiguous();
  //torch::Tensor q_tensor = tt[0];
  //torch::Tensor k_tensor = tt[1];
  //torch::Tensor v_tensor = tt[2];
  rotary_embedding_cuda.SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                 reinterpret_cast<half*>(q_tensor.data_ptr()),
                                 reinterpret_cast<half*>(k_tensor.data_ptr()), total_tokens, stream);
  rotary_embedding_cuda.Forward();

  llm_kernels::nvidia::CacheCopy<half>(reinterpret_cast<half*>(k_tensor.data_ptr()),
                                       reinterpret_cast<half*>(v_tensor.data_ptr()), k_list, v_list,
                                       reinterpret_cast<size_t*>(seqlen), reinterpret_cast<int*>(block_offset),
                                       block_size, batch, total_tokens, num_heads, head_size, stream);
  // flash attention
  flash_attn::mha_varlen_fwd(torch::reshape(q_tensor, {total_tokens, num_heads, head_size}),
                             torch::reshape(k_tensor, {total_tokens, num_heads, head_size}),
                             torch::reshape(tt[2], {total_tokens, num_heads, head_size}), out_tensor,
                             seqlen_tensor.to(torch::kInt32), seqlen_tensor.to(torch::kInt32), max_tokens, max_tokens,
                             0.f, 1.0 / sqrt(head_size), false, is_causal, -1, -1, false, c10::nullopt);
}

template <typename T>
void run_paged_attention(void* out,                // [num_seqs, num_heads, head_size]
                         void* query,              // [num_seqs, num_heads, head_size]
                         void** key_cache_ptrs,    // num_seqs,[seq_blocks]
                         void** value_cache_ptrs,  // num_seqs,[seq_blocks]
                         void* context_lens_ptr,   // [num_seqs]
                         int max_context_len, cudaStream_t stream,
                         void* cache_offsets_ptr,  // num_seqs
                         int num_seqs, int num_heads, int head_size, int num_kv_heads, int block_size, int batch,
                         void* rotary_embedding_pos, int total_tokens,
                         llm_kernels::nvidia::RotaryEmbeddingCuda<half>& rotary_embedding_cuda, void* workspace,
                         size_t work_size, int rank, const std::optional<void*>& alibi_slopes, void* qkv_workspace) {
  const float* alibi_slopes_ptr =
      reinterpret_cast<const float*>(alibi_slopes.has_value() ? alibi_slopes.value() : nullptr);
  auto int64_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt64);

  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat16);
  torch::Tensor qkv_tensor = torch::from_blob(query, {total_tokens, num_heads * head_size * 3}, options);
  auto tt = qkv_tensor.split(qkv_tensor.size(-1) / 3, -1);

  // rotary embedding
  void* q_tensor_ptr = qkv_workspace;
  void* k_tensor_ptr = q_tensor_ptr + (size_t)total_tokens * num_heads * head_size * sizeof(half);
  void* v_tensor_ptr = k_tensor_ptr + (size_t)total_tokens * num_heads * head_size * sizeof(half);

  // copy q k into contiguous GPU memory
  size_t data_size = sizeof(half);
  void* dst = qkv_workspace;
  void* src = query;
  CUDA_CHECK(cudaMemcpy2DAsync((char*)dst + 0 * total_tokens * num_heads * head_size * data_size, num_heads * head_size * data_size,
                               (char*)src + 0 * num_heads * head_size * data_size, 3 * num_heads * head_size * data_size,
                               num_heads * head_size * data_size, total_tokens, cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpy2DAsync((char*)dst + 1 * total_tokens * num_heads * head_size * data_size, num_heads * head_size * data_size,
                               (char*)src + 1 * num_heads * head_size * data_size, 3 * num_heads * head_size * data_size,
                               num_heads * head_size * data_size, total_tokens, cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpy2DAsync((char*)dst + 2 * total_tokens * num_heads * head_size * data_size, num_heads * head_size * data_size,
                               (char*)src + 2 * num_heads * head_size * data_size, 3 * num_heads * head_size * data_size,
                               num_heads * head_size * data_size, total_tokens, cudaMemcpyDeviceToDevice, stream));

  rotary_embedding_cuda.SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                 reinterpret_cast<half*>(q_tensor_ptr),
                                 reinterpret_cast<half*>(k_tensor_ptr), total_tokens, stream);
  rotary_embedding_cuda.Forward();

  llm_kernels::nvidia::CachePosCopy<half>(
      reinterpret_cast<half*>(k_tensor_ptr), reinterpret_cast<half*>(v_tensor_ptr), key_cache_ptrs,
      value_cache_ptrs, rotary_embedding_pos, reinterpret_cast<size_t*>(context_lens_ptr),
      reinterpret_cast<int*>(cache_offsets_ptr), block_size, batch, total_tokens, num_heads, head_size, stream);

  llm_kernels::nvidia::PagedAttentionCuda<uint16_t> op;
  op.SetConfig(num_kv_heads, num_heads, head_size, block_size);

  op.SetInput(reinterpret_cast<uint16_t*>(out), reinterpret_cast<const uint16_t*>(q_tensor_ptr),
              reinterpret_cast<uint16_t**>(key_cache_ptrs), reinterpret_cast<uint16_t**>(value_cache_ptrs),
              reinterpret_cast<const int*>(cache_offsets_ptr), reinterpret_cast<const int*>(context_lens_ptr),
              max_context_len, num_seqs, stream, workspace, work_size, alibi_slopes_ptr);
  op.Forward();
}

template void run_paged_attention<half>(void* out,                // [num_seqs, num_heads, head_size]
                                        void* query,              // [num_seqs, num_heads, head_size]
                                        void** key_cache_ptrs,    // num_seqs,[seq_blocks]
                                        void** value_cache_ptrs,  // num_seqs,[seq_blocks]
                                        void* context_lens_ptr,   // [num_seqs]
                                        int max_context_len, cudaStream_t stream,
                                        void* cache_offsets_ptr,  // num_seqs
                                        int num_seqs, int num_heads, int head_size, int num_kv_heads, int block_size,
                                        int batch, void* rotary_embedding_pos, int total_tokens,
                                        llm_kernels::nvidia::RotaryEmbeddingCuda<half>& rotary_embedding_cuda,
                                        void* workspace, size_t work_size, int rank,
                                        const std::optional<void*>& alibi_slopes, void* qkv_workspace);

void AssembleLastToken(const void* input, const void* offset, const int batch_size, const int hidden_units_num,
                       void* output, cudaStream_t& stream) {
  llm_kernels::nvidia::AssembleLastToken<half>(reinterpret_cast<const half*>(input),
                                               reinterpret_cast<const size_t*>(offset), batch_size, hidden_units_num,
                                               reinterpret_cast<half*>(output), stream);
}

void HalfToFloat(const void* input, const int data_size, void* output, cudaStream_t& stream) {
  llm_kernels::nvidia::HalfToFloat(reinterpret_cast<const half*>(input), data_size, reinterpret_cast<float*>(output),
                                   stream);
}

}  // namespace ksana_llm
