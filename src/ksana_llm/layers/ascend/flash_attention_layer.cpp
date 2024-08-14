/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_attention_layer.h"
#include <cstdlib>
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/kernels/ascend/rotary_embedding/rotary_embedding.h"
#include "csrc/utils/ascend/common.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
Status FlashAttentionLayer<SCALAR_T, CACHE_T, FP8_E5M2>::Init(const std::vector<std::any>& parameters,
                                                              std::shared_ptr<Context> context, int rank) {
#ifdef ENABLE_ACL_ATB
  AttentionLayer<SCALAR_T>::Init(parameters, context, rank);
  if (atb_flash_attn_ == nullptr) {
    atb_flash_attn_ = std::make_shared<llm_kernels::ascend::ATBPagedAttention<SCALAR_T>>();
    atb_flash_attn_->Initialize(static_cast<uint32_t>(this->max_batch_size_), this->num_heads_, this->num_kv_heads_,
                                this->head_size_, this->layer_num_, this->layer_index_, this->block_token_num_,
                                context->GetComputeStreams()[rank].Get(), rank, /*is_context_stage*/ true,
                                this->max_position_embeddings_, this->base_);
  }
  return Status();
#else
  return AttentionLayer<SCALAR_T>::Init(parameters, context, rank);
#endif
}

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
Status FlashAttentionLayer<SCALAR_T, CACHE_T, FP8_E5M2>::Forward(const std::vector<Tensor>& input_tensors,
                                                                 std::vector<Tensor>& output_tensors) {
  // for ACLNN input_tensors:
  //   0: qkv_tensor shape [max_token_num, (2*kv_head_num + head_num)*head_dim], type same as weight
  //   1: input offset tensor shape [max_batch_size + 1], type uint64
  //   2: kv_list shape [num_layer, max_block_num, 2], type pointer
  //   3: prefix_offset_tensor shape [max_batch_size + 1], type int32
  //   4: kv_cache_offset_tensor shape [max_batch_size + 1], type int32
  //   5: rotary embedding pos tensor shape [max_token_num], type int64
  //   6: rotary embedding mask tensor shape [max_token_num], type int64
  //   7: forward shape: [batch_size, max_tokens, kv_cache_offset_list.back()]
  // for ATB input_tensors:
  //   0: qkv_tensor shape [max_token_num, hidden_units * 3], type same as weight
  //   1: slot_mapping shape [max_token_num], type int32_t
  //   2: k_cache shape [max_block_num, block_token_num, head_size, head_dim], type same as weight
  //   3: v_cache shape [max_block_num, block_token_num, head_size, head_dim], type same as weight
  //   4: seq_len_host shape [batch_size]
  //   5: forward shape: [batch_size, max_tokens, kv_cache_offset_list.back()]
  //   6: atb_attention_attr shape: [2], content: 0: layers_slot_mapping_dim_1; 1: max_num_blocks_per_query
  // output_tensors:
  //   0: flash_attention_output shape: [std::max(max_batch_size * vocab_size, max_token_num * hidden_units * 3)]
  void* output = output_tensors[0].GetPtr<void>();
  output_tensors[0].shape[0] = input_tensors[0].shape[0];
  output_tensors[0].shape[1] = this->num_heads_ * this->head_size_;
  output_tensors[0].dtype = input_tensors[0].dtype;
  void* qkv_tensor = input_tensors[0].GetPtr<void>();
  int64_t batch_size = input_tensors[7].shape[0];
  int64_t hidden_units = input_tensors[0].shape[1] / 3;

  void* seq_offset = input_tensors[1].GetPtr<void>();
  void* kv_list = input_tensors[2].GetPtr<void>();
  void* block_offset = input_tensors[4].GetPtr<void>();
  void* rope_pos = input_tensors[5].GetPtr<void>();
  int total_token_num = input_tensors[0].shape[0];
  int total_block_num = input_tensors[7].shape[2];
  this->ascend_paged_attn_->Forward(output, qkv_tensor, seq_offset, reinterpret_cast<void**>(kv_list), block_offset,
                                    rope_pos, batch_size, total_token_num, total_block_num, this->layer_index_, true,
                                    this->context_->GetComputeStreams()[this->rank_].Get());
  return Status();
}
template class FlashAttentionLayer<float, float, false>;
template class FlashAttentionLayer<float, uint8_t, true>;
template class FlashAttentionLayer<float16, float16, false>;
template class FlashAttentionLayer<float16, uint8_t, true>;

}  // namespace ksana_llm
