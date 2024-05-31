/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_attention_layer.h"
#include <cstdlib>
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/kernels/ascend/attention/attention.h"
#include "csrc/kernels/ascend/rotary_embedding/rotary_embedding.h"
#include "csrc/utils/ascend/common.h"

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

  int64_t batch_size = input_tensors[0].shape[0];
  int64_t seq_len = input_tensors[0].shape[1];
  int64_t hidden_units = input_tensors[0].shape[2] / 3;

  int64_t token_pos = seq_len;

  std::vector<int>& padded_token_size = GetPaddedTokenSize();

  void* qkv_ptr = input_tensors[0].GetPtr<void>();
  for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
    int padded_size = padded_token_size.empty() ? 0 : padded_token_size[b_idx];
    int64_t b_seq_len = seq_len - padded_size;

    void* b_qkv_ptr =
        qkv_ptr + ((b_idx * seq_len + padded_size) * (hidden_units * 3 * GetTypeSize(input_tensors[0].dtype)));

    std::vector<int64_t> b_shape = {1, b_seq_len, hidden_units * 3};

    aclTensor* b_input_tensor2;
    llm_kernels::utils::CreateAclTensorWithData(b_shape, &b_qkv_ptr, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                                &b_input_tensor2);

    int b_kvcache_size = input_tensors[11].GetTotalBytes() / batch_size;
    void* b_key_cache2 = input_tensors[11].GetPtr<void>() + (b_idx * b_kvcache_size);
    void* b_val_cache2 = input_tensors[12].GetPtr<void>() + (b_idx * b_kvcache_size);

    std::vector<void*> b_tmp_buffers2;
    int b_tmp_buffer_size = input_tensors[6].GetTotalBytes() / batch_size;
    b_tmp_buffers2.push_back(input_tensors[6].GetPtr<void>() + (b_idx * b_tmp_buffer_size));
    b_tmp_buffers2.push_back(input_tensors[7].GetPtr<void>() + (b_idx * b_tmp_buffer_size));
    b_tmp_buffers2.push_back(input_tensors[8].GetPtr<void>() + (b_idx * b_tmp_buffer_size));
    b_tmp_buffers2.push_back(input_tensors[9].GetPtr<void>() + (b_idx * b_tmp_buffer_size));
    b_tmp_buffers2.push_back(input_tensors[10].GetPtr<void>() + (b_idx * b_tmp_buffer_size));

    size_t b_workspace_needed = b_seq_len * hidden_units * sizeof(uint16_t) * 3;
    void* b_workspace_buf_ptr = nullptr;
    AttentionLayer<T>::PrepareWorkspaceBuffer(b_workspace_needed, b_workspace_buf_ptr);

    int64_t b_token_pos = b_seq_len;

    aclTensor* b_output2;
    this->ascend_flash_attn_->Forward(b_input_tensor2, b_token_pos, &b_key_cache2, &b_val_cache2, b_tmp_buffers2,
                                      &b_output2, true, context_->GetComputeStreams()[rank_].Get(), GetWorkSpaceFunc(),
                                      b_workspace_buf_ptr);

    size_t b_size2 = b_seq_len * hidden_units * GetTypeSize(input_tensors[0].dtype);
    ACL_CHECK(
        aclrtMemcpyAsync(output_tensors[0].GetPtr<void>() +
                             ((b_idx * seq_len + padded_size) * (hidden_units * GetTypeSize(input_tensors[0].dtype))),
                         b_size2, b_tmp_buffers2[1], b_size2, aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE,
                         context_->GetComputeStreams()[rank_].Get()));

    ACL_CHECK(aclDestroyTensor(b_input_tensor2));
  }

  output_tensors[0].shape = {static_cast<unsigned long>(batch_size), static_cast<unsigned long>(seq_len),
                             static_cast<unsigned long>(hidden_units)};
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}
template class FlashAttentionLayer<float>;
template class FlashAttentionLayer<float16>;

}  // namespace ksana_llm
