/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/kernels/nvidia/remove_me_later.h"
#include "csrc/kernels/nvidia/embedding/embedding.h"


#include <iostream>

namespace numerous_llm {

void JustPlaceholder::Hello() { std::cout << "Please remove me later." << std::endl; }
// kernel host代码代补充

//template <typename T>
//void LookupFusedEmbeddingWithCSRInputs(T* output_hidden_units, const T* embedding_table, const T* pos_table,
//                                       InvokeInputIdsEmbeddingLookupPosEncodingParam<T> prompt_param,
//                                       const int32_t* input_ids, const int start_step, const size_t* ids_offsets,
//                                       const int batch_size, const uint32_t hidden_units, const size_t vocab_size,
//                                       const size_t vocab_id, cudaStream_t stream);
void emb_lookup(const void* ids, const void* offset, const void* emb, const void* pos, void* output, int vocab_size, int hidden_size, int bs, int step, int vocab_id, cudaStream_t stream) {
  llm_kernels::nvidia::LookupFusedEmbeddingWithCSRInputs<half>(
      reinterpret_cast<half*>(output), reinterpret_cast<const half*>(emb), reinterpret_cast<const half*>(pos), {},
      reinterpret_cast<const int32_t*>(ids), step, reinterpret_cast<const size_t*>(offset), bs, hidden_size, vocab_size, vocab_id, stream);
}
void layernorm(const void* input, const void* weight, void* output, cudaStream_t stream) {

}
}  // namespace numerous_llm
