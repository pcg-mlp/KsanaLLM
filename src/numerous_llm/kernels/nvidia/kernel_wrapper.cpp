/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/kernels/nvidia/kernel_wrapper.h"
#include "csrc/kernels/nvidia/embedding/embedding.h"
#include "csrc/kernels/nvidia/layernorm/layernorm.h"

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
  llm_kernels::nvidia::invokeLayerNorm<half>(reinterpret_cast<half*>(output), reinterpret_cast<const half*>(input),
                                             reinterpret_cast<const half*>(weight), beta, layernorm_eps, m, n, stream);
}

}  // namespace numerous_llm
