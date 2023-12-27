/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/flash_attention_layer.h"
#include "numerous_llm/kernels/nvidia/kernel_wrapper.h"

namespace numerous_llm {

// kernel host代码代补充
void flash_attention(const int layer_index, const Tensor& input, Tensor output, std::vector<Tensor>& key_cache,
                     std::vector<Tensor>& value_cache, int max_position, cudaStream_t stream) {}

Status FlashAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  //Tensor& out = output_tensors[0];
  //int cache_len = (output_tensors.size() - 1) / 2;
  //std::vector<Tensor> key_cache(output_tensors.begin() + 1, output_tensors.begin() + 1 + cache_len);
  //std::vector<Tensor> value_cache(output_tensors.begin() + 1 + cache_len, output_tensors.begin() + 1 + cache_len * 2);
  //flash_attention(layer_index_, input_tensors[0], out, key_cache, value_cache, max_position_embeddings_, stream_);
  return Status();
  int total_tokens;
  int max_tokens;
  int batch;
  int num_heads; // init
  int head_size; // init
  bool is_causal; // init
  int rank; // init
  AttenVarlen(input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(), output_tensors[0].GetPtr<void>(), input_tensors[3].GetPtr<void>(), 
              total_tokens,  max_tokens,  batch, num_heads,  head_size,  is_causal,  rank_, context_->GetComputeStreams()[rank_]);
  return Status();
}

}  // namespace numerous_llm
