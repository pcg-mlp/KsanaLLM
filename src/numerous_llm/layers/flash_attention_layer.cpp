/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/flash_attention_layer.h"
#include "numerous_llm/kernels/nvidia/kernel_wrapper.h"

namespace numerous_llm {

Status FlashAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int max_tokens = input_tensors[1].shape[0];
  int batch_size = 1;
  int total_tokens = input_tensors[0].shape[0];

  NLLM_LOG_INFO << fmt::format("max_tokens = {}, batch_size = {}, total_tokens = {}",
                               max_tokens, batch_size, total_tokens);

  size_t qkv_size = input_tensors[0].GetTotalBytes();
  NLLM_LOG_INFO << fmt::format("qkv bytes size = {}", qkv_size);
  void* qkv_ptr = input_tensors[0].GetPtr<void>();

  void* q_ptr = qkv_ptr;
  void* k_ptr = qkv_ptr + qkv_size / 3;
  void* v_ptr = qkv_ptr + qkv_size / 3 * 2;

  AttenVarlen(q_ptr, k_ptr, v_ptr, output_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), total_tokens,
              max_tokens, batch_size, num_heads_, head_size_, is_causal_, rank_, context_->GetComputeStreams()[rank_]);
  output_tensors[0].shape[0] = input_tensors[0].shape[0];
  output_tensors[0].shape[1] = input_tensors[0].shape[1] / 3;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace numerous_llm
