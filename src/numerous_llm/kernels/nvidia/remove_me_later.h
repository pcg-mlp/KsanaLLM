/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cuda_runtime.h>
namespace numerous_llm {

class JustPlaceholder {
 public:
  void Hello();
};
void emb_lookup(const void* ids, const void* offset, const void* emb, const void* pos, void* output, int vocab_size, int hidden_size, int bs, int step, int vocab_id, cudaStream_t stream);
void layernorm(const void* input, const void* weight, void* output, cudaStream_t stream);

}  // namespace numerous_llm
