/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/paged_attention_layer.h"
#include "numerous_llm/kernels/nvidia/paged_attention.h"

namespace numerous_llm {
Status PagedAttentionLayer::Init(const std::vector<std::any>& parameters, cudaStream_t stream) {
  stream_ = stream;
  rotary_embedding_layer_.Init(parameters, stream);
  return Status();
}

Status PagedAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status();

  // TODO: fixed invoke error
  Tensor& out = output_tensors[0];
  int cache_len = (output_tensors.size() - 1) / 2;
  std::vector<Tensor> key_cache(output_tensors.begin() + 1, output_tensors.begin() + 1 + cache_len);
  std::vector<Tensor> value_cache(output_tensors.begin() + 1 + cache_len, output_tensors.begin() + 1 + cache_len * 2);
  paged_attention(out, input_tensors[0], key_cache, value_cache, input_tensors[1], 0, stream_, nullptr, 0, {});
  std::vector<Tensor> rotary_embedding_layer_input_and_output = {out};
  rotary_embedding_layer_.Forward(rotary_embedding_layer_input_and_output, rotary_embedding_layer_input_and_output);
  return Status();
}

}  // namespace numerous_llm
