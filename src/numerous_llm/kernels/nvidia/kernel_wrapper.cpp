/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/kernels/nvidia/kernel_wrapper.h"
#include "csrc/kernels/nvidia/add/add.h"
#include "csrc/kernels/nvidia/embedding/embedding.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "csrc/kernels/nvidia/layernorm/layernorm.h"

#include "numerous_llm/utils/nvidia/cuda_utils.h"

#include <iostream>

namespace numerous_llm {

// kernel host代码代补充

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

}  // namespace numerous_llm
