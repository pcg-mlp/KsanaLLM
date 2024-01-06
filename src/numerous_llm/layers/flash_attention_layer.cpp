/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/flash_attention_layer.h"
#include "numerous_llm/kernels/nvidia/kernel_wrapper.h"

namespace numerous_llm {

Status FlashAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  /*
   * input_tensors:
   *    0: qkv_tensor
   *    1: input offset tensor (uint64)
   *    2: kv_list
   *    3: kv_cache_buffer 连续大空间
   *    4: rotary embedding pos tensor
   *    5: forward shape
  */

  int max_tokens = input_tensors[5].shape[1];
  int batch_size = input_tensors[5].shape[0];
  int total_tokens = input_tensors[0].shape[0];

  NLLM_LOG_INFO << fmt::format("max_tokens = {}, batch_size = {}, total_tokens = {}",
                               max_tokens, batch_size, total_tokens);

  AttenVarlen(input_tensors[0].GetPtr<void>(), input_tensors[4].GetPtr<void>(), output_tensors[0].GetPtr<void>(),
              input_tensors[1].GetPtr<void>(), rotary_embedding_cuda_, total_tokens, max_tokens, batch_size,
              num_heads_, head_size_, is_causal_, rank_, context_->GetComputeStreams()[rank_]);
  output_tensors[0].shape[0] = input_tensors[0].shape[0];
  output_tensors[0].shape[1] = input_tensors[0].shape[1] / 3;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace numerous_llm
