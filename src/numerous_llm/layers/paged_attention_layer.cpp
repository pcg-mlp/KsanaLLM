/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/paged_attention_layer.h"
#include "numerous_llm/kernels/nvidia/paged_attention.h"

namespace numerous_llm {
Status PagedAttentionLayer::Init(const std::vector<std::any>& parameters, cudaStream_t stream) {
  AttentionLayer::Init(parameters, stream);
  rotary_embedding_layer_.Init({max_position_embeddings_}, stream);
  return Status();
}

// input_tensors = {query, kv_list, context_lens, workerspace}
Status PagedAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  const Tensor& query = input_tensors[0];
  const Tensor& context_lens = input_tensors[1];
  Tensor& out = output_tensors[0];
  Tensor& kv_list = output_tensors[1];
  Tensor& workspace = output_tensors[2];
  paged_attention(layer_index_, out, query, kv_list, block_size_, context_lens, 0, stream_, workspace, {});
  std::vector<Tensor> rotary_embedding_layer_input_and_output = {out};
  rotary_embedding_layer_.Forward(rotary_embedding_layer_input_and_output, rotary_embedding_layer_input_and_output);
  return Status();
}

}  // namespace numerous_llm
