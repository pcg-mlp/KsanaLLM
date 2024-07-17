/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_attention_layer.h"
#include <cstdlib>
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/kernels/ascend/rotary_embedding/rotary_embedding.h"
#include "csrc/utils/ascend/common.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
Status FlashAttentionLayer<SCALAR_T, CACHE_T, FP8_E5M2>::Forward(const std::vector<Tensor>& input_tensors,
                                                                 std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //     0: qkv_tensor shape [max_token_num, hidden_units, 3], type same as weight
  //     1: input offset tensor shape [max_batch_size + 1], type uint64
  //     2: kv_list shape [num_layer, max_block_num, 2], type pointer
  //     3: prefix_offset_tensor shape [max_batch_size + 1], type int32
  //     4: kv_cache_offset_tensor shape [max_batch_size + 1], type int32
  //     5: rotary embedding pos tensor shape [max_token_num], type int64
  //     6: rotary embedding mask tensor shape [max_token_num], type int64
  //     7: forward shape: [batch_size, max_tokens, kv_cache_offset_list.back()]
  //     8~12: ascend buffers: [max_token_num, hidden_units]
  //     13~14: ascend kvcache buffers: [max_token_num, hidden_units]
  // output_tensors:
  //     0: flash_attention_output shape: [std::max(max_batch_size * vocab_size, max_token_num * hidden_units * 3)]

  int64_t batch_size = input_tensors[7].shape[0];
  int64_t hidden_units = input_tensors[0].shape[1] / 3;

  void* output = output_tensors[0].GetPtr<void>();
  void* qkv_tensor = input_tensors[0].GetPtr<void>();
  void* seq_offset = input_tensors[1].GetPtr<void>();
  void* kv_list = input_tensors[2].GetPtr<void>();
  void* block_offset = input_tensors[4].GetPtr<void>();
  void* rope_pos = input_tensors[5].GetPtr<void>();

  int total_token_num = input_tensors[0].shape[0];
  int total_block_num = input_tensors[7].shape[2];

  this->ascend_paged_attn_->Forward(output, qkv_tensor, seq_offset, reinterpret_cast<void**>(kv_list), block_offset,
                                    rope_pos, batch_size, total_token_num, total_block_num, this->layer_index_, true,
                                    this->context_->GetComputeStreams()[this->rank_].Get());

  output_tensors[0].shape[0] = input_tensors[0].shape[0];
  output_tensors[0].shape[1] = this->num_heads_ * this->head_size_;
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}
template class FlashAttentionLayer<float, float, false>;
template class FlashAttentionLayer<float, uint8_t, true>;
template class FlashAttentionLayer<float16, float16, false>;
template class FlashAttentionLayer<float16, uint8_t, true>;

}  // namespace ksana_llm
