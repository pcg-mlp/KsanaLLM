/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/emb_lookup_layer.h"

namespace numerous_llm {
// kernel host代码代补充
void emb_lookup(const Tensor& input, const Tensor& weight, Tensor output, cudaStream_t stream) {}

Status EmbLookupLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  emb_lookup(input_tensors[0], input_tensors[1], output_tensors[0], stream_);
  return Status();
}
}  // namespace numerous_llm
