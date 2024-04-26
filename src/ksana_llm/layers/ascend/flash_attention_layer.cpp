/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_attention_layer.h"
#include <cstdlib>
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/kernels/ascend/attention/attention.h"
#include "csrc/kernels/ascend/rotary_embedding/rotary_embedding.h"

namespace ksana_llm {

template <typename T>
Status FlashAttentionLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //     0: qkv_tensor shape [max_token_num, hidden_units, 3], type same as weight
  //     1: input offset tensor shape [max_batch_size + 1], type uint64
  //     2: kv_list shape [num_layer, max_block_num, 2], type pointer
  //     3: kv_cache_offset_tensor shape [max_batch_size + 1], type int32
  //     4: rotary embedding pos tensor shape [max_token_num], type int64
  //     5: forward shape: [batch_size, max_tokens, kv_cache_offset_list.back()]
  //     6~10: ascend buffers: [max_token_num, hidden_units]
  //     11~12: ascend kvcache buffers: [max_token_num, hidden_units]
  // output_tensors:
  //     0: flash_attention_output shape: [std::max(max_batch_size * vocab_size, max_token_num * hidden_units * 3)]

  int64_t seq_len = input_tensors[5].shape[1];
  int64_t token_pos = seq_len;
  int64_t hidden_units = input_tensors[0].shape[1] / 3;

  // shape: [bs, seq_len, hidden_units, 3]
  Tensor qkv_input2 = input_tensors[0];
  aclTensor* input_tensor2 = qkv_input2.ResetDeviceTensor(DataType::TYPE_FP16, {1, seq_len, hidden_units * 3});

  void* key_cache2 = input_tensors[11].GetPtr<void>();
  void* val_cache2 = input_tensors[12].GetPtr<void>();

  std::vector<void*> tmp_buffers2;
  tmp_buffers2.push_back(input_tensors[6].GetPtr<void>());
  tmp_buffers2.push_back(input_tensors[7].GetPtr<void>());
  tmp_buffers2.push_back(input_tensors[8].GetPtr<void>());
  tmp_buffers2.push_back(input_tensors[9].GetPtr<void>());
  tmp_buffers2.push_back(input_tensors[10].GetPtr<void>());

  size_t workspace_needed = seq_len * hidden_units * sizeof(uint16_t) * 3;
  void* workspace_buf_ptr = nullptr;
  AttentionLayer<T>::PrepareWorkspaceBuffer(workspace_needed, workspace_buf_ptr);

  aclTensor* output2;
  this->ascend_flash_attn_->Forward(input_tensor2, token_pos, &key_cache2, &val_cache2, tmp_buffers2, &output2, true,
                                    context_->GetComputeStreams()[rank_].Get(), GetWorkSpaceFunc(), workspace_buf_ptr);

  size_t size2 = seq_len * hidden_units * GetTypeSize(input_tensors[0].dtype);
  ACL_CHECK(aclrtMemcpy(output_tensors[0].GetPtr<void>(), size2, tmp_buffers2[1], size2,
                        aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE));

  output_tensors[0].shape = {static_cast<unsigned long>(seq_len), static_cast<unsigned long>(hidden_units)};
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}
template class FlashAttentionLayer<float>;
template class FlashAttentionLayer<float16>;

}  // namespace ksana_llm
