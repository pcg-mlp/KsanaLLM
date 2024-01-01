/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/kernels/nvidia/kernel_wrapper.h"

#include <fstream>
#include <iostream>

#include "flash_api.h"

#include "csrc/kernels/nvidia/activation/activation.h"
#include "csrc/kernels/nvidia/add/add.h"
#include "csrc/kernels/nvidia/assemble_last_token/assemble_last_token.h"
#include "csrc/kernels/nvidia/embedding/embedding.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "csrc/kernels/nvidia/layernorm/layernorm.h"

#include "numerous_llm/utils/nvidia/cuda_utils.h"

namespace numerous_llm {

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

void InvokeAddBiasResidual(const void* input, const void* bias, const int m, const int n, void* output,
                           cudaStream_t stream) {
  llm_kernels::nvidia::InvokeAddBiasResidual<half>(reinterpret_cast<half*>(output),
                                                   reinterpret_cast<const half*>(input),
                                                   reinterpret_cast<const half*>(bias), m, n, stream);
}

void InvokeSiluActivation(const void* input, const void* bias, const int m, const int n, void* output,
                          cudaStream_t stream) {
  const int* ia3_tasks = nullptr;
  const half* ia3_weights = nullptr;
  const half* gated_weights = nullptr;
  const half* gated_bias = nullptr;
  const int int8_mode = 0;
  const int* padding_offset = nullptr;
  const int seq_len = 0;
  const float* activation_in = nullptr;
  const float* activation_out = nullptr;

  CUDA_CHECK(cudaMemcpyAsync(output, input, sizeof(half) * m * n, cudaMemcpyDeviceToDevice, stream));
  llm_kernels::nvidia::InvokeGenericActivation<llm_kernels::nvidia::SiluActivation, half, half>(
      reinterpret_cast<half*>(output), reinterpret_cast<const half*>(bias), gated_weights, gated_bias, ia3_tasks,
      ia3_weights, m, n, int8_mode, activation_in, activation_out, padding_offset, seq_len, stream);
}

void AttenVarlen(void* q, void* k, void* v, void* out, void* seqlen, int total_tokens, int max_tokens, int batch,
                 int num_heads, int head_size, bool is_causal, int rank, cudaStream_t stream) {
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat16);
  torch::Tensor q_tensor = torch::from_blob(q, {total_tokens, num_heads, head_size}, options);
  torch::Tensor k_tensor = torch::from_blob(k, {total_tokens, num_heads, head_size}, options);
  torch::Tensor v_tensor = torch::from_blob(v, {total_tokens, num_heads, head_size}, options);
  c10::optional<at::Tensor> out_tensor = torch::from_blob(out, {total_tokens, num_heads, head_size}, options);

  auto int_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt32);
  torch::Tensor seqlen_tensor = torch::from_blob(seqlen, {batch + 1}, int_options);

  flash_attn::mha_varlen_fwd(q_tensor, k_tensor, v_tensor, out_tensor, seqlen_tensor, seqlen_tensor, max_tokens,
                             max_tokens, 0.f, 1.0 / sqrt(num_heads), false, is_causal, -1, -1, false, c10::nullopt);
}

void AssembleLastToken(const void* input, const void* offset, const int batch_size, const int hidden_units_num,
                       void* output, cudaStream_t& stream) {
  llm_kernels::nvidia::AssembleLastToken<half>(reinterpret_cast<const half*>(input),
                                               reinterpret_cast<const size_t*>(offset), batch_size, hidden_units_num,
                                               reinterpret_cast<half*>(output), stream);
}

}  // namespace numerous_llm
