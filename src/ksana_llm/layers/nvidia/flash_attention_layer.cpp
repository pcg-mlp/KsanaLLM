/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_attention_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
Status FlashAttentionLayer<SCALAR_T, CACHE_T, FP8_E5M2>::Init(const std::vector<std::any>& parameters,
                                                              std::shared_ptr<Context> context, int rank) {
  return AttentionLayer<SCALAR_T>::Init(parameters, context, rank);
}

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
Status FlashAttentionLayer<SCALAR_T, CACHE_T, FP8_E5M2>::Forward(const std::vector<Tensor>& input_tensors,
                                                                 std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //     0: qkv_tensor shape [total_token_num, hidden_units * 3], type same as weight
  //     1: token_offset tensor shape [max_batch_size + 1], type uint64
  //     2: kv_list shape [num_layer, max_block_num, 2], type pointer
  //     3: prefix_offset_tensor shape [max_batch_size + 1], type int32
  //     4: kv_cache_offset_tensor shape [max_batch_size + 1], type int32
  //     5: rotary embedding pos tensor shape [max_token_num], type int64
  //     6: rotary embedding mask tensor shape [max_token_num], type int64
  //     7: forward shape: [batch_size, max_tokens, kv_cache_offset_list.back()]
  // output_tensors:
  //     0: flash_attention_output shape: [std::max(max_batch_size * vocab_size, max_token_num * hidden_units * 3)]

  int max_tokens = input_tensors[7].shape[1];
  int batch_size = input_tensors[7].shape[0];
  int layer_block_num = input_tensors[7].shape[2];
  int total_tokens = input_tensors[0].shape[0];

  void** k_list = (input_tensors[2].GetPtr<void*>()) + this->layer_index_ * layer_block_num * 2;
  void** v_list = k_list + layer_block_num;
  AttenVarlen<SCALAR_T, CACHE_T, FP8_E5M2>(
      input_tensors[0].GetPtr<void>(), input_tensors[5].GetPtr<void>(), input_tensors[6].GetPtr<void>(),
      output_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), this->rotary_embedding_cuda_, total_tokens,
      max_tokens, batch_size, this->num_heads_, this->num_kv_heads_, this->head_size_, this->stride_size_,
      this->tensor_para_size_, this->is_causal_, this->rank_, this->block_token_num_, k_list, v_list,
      input_tensors[3].GetPtr<void>(), input_tensors[4].GetPtr<void>(), this->alibi_slopes_,
      this->context_->GetComputeStreams()[this->rank_].Get());
  output_tensors[0].shape[0] = input_tensors[0].shape[0];
  output_tensors[0].shape[1] = this->num_heads_ * this->head_size_;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

template class FlashAttentionLayer<float, float, false>;
template class FlashAttentionLayer<float, uint8_t, true>;
template class FlashAttentionLayer<half, half, false>;
template class FlashAttentionLayer<half, uint8_t, true>;
#ifdef ENABLE_BFLOAT16
template class FlashAttentionLayer<__nv_bfloat16, __nv_bfloat16, false>;
template class FlashAttentionLayer<__nv_bfloat16, uint8_t, true>;
#endif

}  // namespace ksana_llm
