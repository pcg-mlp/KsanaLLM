/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/flash_attention_layer.h"
#include "numerous_llm/kernels/nvidia/kernel_wrapper.h"

namespace numerous_llm {

Status FlashAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int total_tokens = input_tensors[0].shape[0];
  int max_tokens = input_tensors[0].shape[0];
  int batch_size = input_tensors[3].shape[0];
  AttenVarlen(input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), input_tensors[1].GetPtr<void>(),
              output_tensors[0].GetPtr<void>(), input_tensors[3].GetPtr<void>(), total_tokens, max_tokens, batch_size,
              num_heads_, head_size_, is_causal_, rank_, context_->GetComputeStreams()[rank_]);
  return Status();
}

}  // namespace numerous_llm
