/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_attention_layer.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

namespace ksana_llm {

Status FlashAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //     0: qkv_tensor shape [max_token_num, hidden_units, 3], type same as weight
  //     1: input offset tensor shape [max_batch_size + 1], type uint64
  //     2: kv_list shape [num_layer, max_block_num, 2], type pointer
  //     3: kv_cache_offset_tensor shape [max_batch_size + 1], type int32
  //     4: rotary embedding pos tensor shape [max_token_num], type int64
  //     5: forward shape: [batch_size, max_tokens, kv_cache_offset_list.back()]
  // output_tensors:
  //     0: flash_attention_output shape: [std::max(max_batch_size * vocab_size, max_token_num * hidden_units * 3)]

  // TODO(karlluo): implement llm_kernels::ascend::FlashAttention
  return Status();
}

}  // namespace ksana_llm
