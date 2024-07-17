/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/kernels/ascend/rotary_embedding/rotary_embedding.h"
#include "csrc/utils/ascend/common.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
Status PagedAttentionLayer<SCALAR_T, CACHE_T, FP8_E5M2>::Forward(const std::vector<Tensor>& input_tensors,
                                                                 std::vector<Tensor>& output_tensors) {
  // PagedAttention部分
  // input_tensors:
  //   0: 输入数据
  //   1: int_input_tokens_tensor
  //   2: kv_list
  //   3: kv_cache_offset_tensor
  //   4: rotary_embedding_pos
  //   5: rotary_embedding_mask
  //   6: workspace 空间
  //   7: forward_shape
  //   8: 用于存储 qk 的临时空间(TODO:)
  //   9~13: ascend buffers: [max_token_num, hidden_units]
  //   14~15: ascend kvcache buffers: [max_token_num, hidden_units]
  // output_tensors:
  //   0: paged attention output
  // NLLM_LOG_WARNING <<"";
  // 块的位移情况
  // 如上kv_list的情况
  // 一块有8个token时
  // context_lens是17,41
  // input_offse是0,17,58
  // cache_offset是0,3,9

  int64_t batch_size = input_tensors[7].shape[0];
  int64_t hidden_units = input_tensors[0].shape[1] / 3;

  int total_token_num = input_tensors[0].shape[0];
  int total_block_num = input_tensors[7].shape[2];

  void* output = output_tensors[0].GetPtr<void>();
  void* qkv_tensor = input_tensors[0].GetPtr<void>();
  void* seq_offset = input_tensors[1].GetPtr<void>();
  void* kv_list = input_tensors[2].GetPtr<void>();
  void* block_offset = input_tensors[3].GetPtr<void>();
  void* rope_pos = input_tensors[4].GetPtr<void>();

  this->ascend_paged_attn_->Forward(output, qkv_tensor, seq_offset, reinterpret_cast<void**>(kv_list), block_offset,
                                    rope_pos, batch_size, total_token_num, total_block_num, this->layer_index_, false,
                                    this->context_->GetComputeStreams()[this->rank_].Get());

  output_tensors[0].shape = {static_cast<uint64_t>(batch_size), static_cast<uint64_t>(hidden_units)};
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}
template class PagedAttentionLayer<float, float, false>;
template class PagedAttentionLayer<float, uint8_t, true>;
template class PagedAttentionLayer<float16, float16, false>;
template class PagedAttentionLayer<float16, uint8_t, true>;

}  // namespace ksana_llm
