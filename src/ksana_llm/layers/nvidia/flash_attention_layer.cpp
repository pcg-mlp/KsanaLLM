/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_attention_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Init(const std::vector<std::any>& parameters,
                                                              std::shared_ptr<Context> context, int rank) {
  return AttentionLayer<SCALAR_T>::Init(parameters, context, rank);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Forward(const std::vector<Tensor>& input_tensors,
                                                                 std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //     0: qkv_tensor shape [total_token_num, hidden_units * 3], type same as weight
  //     1: token_offset tensor shape [max_batch_size + 1], type uint64
  //     2: kv_list shape [num_layer, max_block_num, 2], type pointer
  //     3: prefix_offset_tensor shape [max_batch_size + 1], type int32
  //     4: kv_cache_offset_tensor shape [max_batch_size + 1], type int32
  //     5: rotary embedding pos tensor shape [max_token_num], type int64
  //     6: rotary embedding mask tensor shape [max_token_num], type int64
  //     7: forward shape: [batch_size, max_tokens, kv_cache_offset_list.back(), xx, xx, xx, without_prefix_max_tokens]
#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
  //     8: kv_cache_base_ptr_tensor: [1 + layer_num * 2]
  //     9: block_table: [batch_size, max_tokens]
  //     10: input_without_prefix_offset_tensor: [batch_size + 1]
#endif
  // output_tensors:
  //     0: flash_attention_output shape: [std::max(max_batch_size * vocab_size, max_token_num * hidden_units * 3)]

  int max_tokens = input_tensors[7].shape[1];
  int batch_size = input_tensors[7].shape[0];
  int layer_block_num = input_tensors[7].shape[2];
  int total_tokens = input_tensors[0].shape[0] - input_tensors[7].shape[3];

  void** k_list = (input_tensors[2].GetPtr<void*>()) + this->layer_index_ * layer_block_num * 2;
  void** v_list = k_list + layer_block_num;

#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
  int64_t kv_cache_block_num = *(input_tensors[8].GetPtr<int64_t>());
  void** layer_kv_cache_ptr = input_tensors[8].GetPtr<void*>() + 1;
  void* k_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2];
  void* v_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2 + 1];
  int32_t* block_table_ptr = input_tensors[9].GetPtr<int32_t>();
  int max_blocks_per_seq = input_tensors[9].shape[1];
  size_t* input_without_prefix_offset = input_tensors[10].GetPtr<size_t>();
  int without_prefix_max_tokens = input_tensors[7].shape[6];
#endif

  AttenVarlen<SCALAR_T, CACHE_T, KV_DTYPE>(
      input_tensors[0].GetPtr<void>(), input_tensors[5].GetPtr<void>(), input_tensors[6].GetPtr<void>(),
      output_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), this->rotary_embedding_cuda_, total_tokens,
      max_tokens, batch_size, this->num_heads_, this->num_kv_heads_, this->head_size_, this->stride_size_,
      this->k_scale_, this->v_scale_, this->tensor_para_size_, this->is_causal_, this->rank_, this->block_token_num_,
      k_list, v_list, input_tensors[3].GetPtr<void>(), input_tensors[4].GetPtr<void>(), this->alibi_slopes_,
#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
      this->context_->GetComputeStreams()[this->rank_].Get(), k_cache_ptr, v_cache_ptr, block_table_ptr,
      kv_cache_block_num, max_blocks_per_seq, input_without_prefix_offset, without_prefix_max_tokens);
#else
      this->context_->GetComputeStreams()[this->rank_].Get());
#endif
  output_tensors[0].shape[0] = total_tokens;
  output_tensors[0].shape[1] = this->num_heads_ * this->head_size_;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

using llm_kernels::utils::KVCacheType;
template class FlashAttentionLayer<float, float, KVCacheType::kAuto>;
template class FlashAttentionLayer<float, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashAttentionLayer<float, uint8_t, KVCacheType::kFp8E5M2>;
template class FlashAttentionLayer<half, half, KVCacheType::kAuto>;
template class FlashAttentionLayer<half, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashAttentionLayer<half, uint8_t, KVCacheType::kFp8E5M2>;
#ifdef ENABLE_BFLOAT16
template class FlashAttentionLayer<__nv_bfloat16, __nv_bfloat16, KVCacheType::kAuto>;
template class FlashAttentionLayer<__nv_bfloat16, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashAttentionLayer<__nv_bfloat16, uint8_t, KVCacheType::kFp8E5M2>;
#endif

}  // namespace ksana_llm
