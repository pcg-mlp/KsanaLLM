/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/kernels/ascend/attention/attention.h"
#include "csrc/kernels/ascend/rotary_embedding/rotary_embedding.h"
#include "csrc/utils/ascend/common.h"

namespace ksana_llm {

template <typename T>
Status PagedAttentionLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
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
  int max_tokens = input_tensors[7].shape[1];

  int64_t seq_len = 1;
  int64_t token_pos = max_tokens - 1;

  int64_t batch_size = input_tensors[0].shape[0];
  int64_t hidden_units = input_tensors[0].shape[2] / 3;

  std::vector<int>& padded_token_size = GetPaddedTokenSize();

  void* qkv_ptr = input_tensors[0].GetPtr<void>();
  for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
    int padded_size = padded_token_size.empty() ? 0 : padded_token_size[b_idx];

    void* b_qkv_ptr = qkv_ptr + (b_idx * hidden_units * 3 * GetTypeSize(input_tensors[0].dtype));

    std::vector<int64_t> b_shape = {1, 1, hidden_units * 3};

    aclTensor* b_input_tensor;
    llm_kernels::utils::CreateAclTensorWithData(b_shape, &b_qkv_ptr, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                                &b_input_tensor);

    int b_kvcache_size = input_tensors[14].GetTotalBytes() / batch_size;
    void* b_key_cache = input_tensors[14].GetPtr<void>() + (b_idx * b_kvcache_size);
    void* b_val_cache = input_tensors[15].GetPtr<void>() + (b_idx * b_kvcache_size);

    std::vector<void*> b_tmp_buffers;
    int b_tmp_buffer_size = input_tensors[9].GetTotalBytes() / batch_size;
    b_tmp_buffers.push_back(input_tensors[9].GetPtr<void>() + (b_idx * b_tmp_buffer_size));
    b_tmp_buffers.push_back(input_tensors[10].GetPtr<void>() + (b_idx * b_tmp_buffer_size));
    b_tmp_buffers.push_back(input_tensors[11].GetPtr<void>() + (b_idx * b_tmp_buffer_size));
    b_tmp_buffers.push_back(input_tensors[12].GetPtr<void>() + (b_idx * b_tmp_buffer_size));
    b_tmp_buffers.push_back(input_tensors[13].GetPtr<void>() + (b_idx * b_tmp_buffer_size));

    size_t b_workspace_needed = hidden_units * sizeof(uint16_t) * 3;
    void* b_workspace_buf_ptr = nullptr;
    AttentionLayer<T>::PrepareWorkspaceBuffer(b_workspace_needed, b_workspace_buf_ptr);

    int64_t b_token_pos = max_tokens - 1 - padded_size;

    aclTensor* b_output;
    this->ascend_flash_attn_->Forward(b_input_tensor, b_token_pos, &b_key_cache, &b_val_cache, b_tmp_buffers, &b_output,
                                      false, context_->GetComputeStreams()[rank_].Get(), GetWorkSpaceFunc(),
                                      b_workspace_buf_ptr);

    size_t b_size = hidden_units * GetTypeSize(input_tensors[0].dtype);
    ACL_CHECK(
        aclrtMemcpy(output_tensors[0].GetPtr<void>() + (b_idx * hidden_units * GetTypeSize(input_tensors[0].dtype)),
                    b_size, b_tmp_buffers[1], b_size, aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE));

    ACL_CHECK(aclDestroyTensor(b_input_tensor));
  }

  output_tensors[0].shape = {static_cast<unsigned long>(batch_size), static_cast<unsigned long>(seq_len),
                             static_cast<unsigned long>(hidden_units)};
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}
template class PagedAttentionLayer<float>;
template class PagedAttentionLayer<float16>;

}  // namespace ksana_llm
