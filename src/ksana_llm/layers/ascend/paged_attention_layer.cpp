/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/kernels/ascend/attention/attention.h"
#include "csrc/kernels/ascend/rotary_embedding/rotary_embedding.h"

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
  //   5: workspace 空间
  //   6: forward_shape
  //   7: 用于存储 qk 的临时空间(TODO:)
  //   8~12: ascend buffers: [max_token_num, hidden_units]
  // output_tensors:
  //   0: paged attention output
  // NLLM_LOG_WARNING <<"";
  const Tensor& query = input_tensors[0];
  const Tensor& context_lens = input_tensors[1];
  // 块的位移情况
  // 如上kv_list的情况
  // 一块有8个token时
  // context_lens是17,41
  // input_offse是0,17,58
  // cache_offset是0,3,9
  const Tensor& kv_list = input_tensors[2];
  const Tensor& cache_offset = input_tensors[3];
  const Tensor& rotary_embedding_pos = input_tensors[4];
  const Tensor& workspace = input_tensors[5];
  const Tensor& forward_shape = input_tensors[6];
  const Tensor& qkv_workspace = input_tensors[7];
  int layer_block_num = input_tensors[6].shape[2];
  int max_tokens = input_tensors[6].shape[1];
  int batch_size = input_tensors[6].shape[0];
  int total_tokens = input_tensors[0].shape[0];

  int64_t seq_len = 1;
  int64_t token_pos = max_tokens - 1;

  int64_t hidden_units = input_tensors[0].shape[1] / 3;

  // shape: [bs, seq_len, hidden_units, 3]
  aclTensor* qkv_tensor;
  Tensor qkv_input = input_tensors[0];
  aclTensor* input_tensor = qkv_input.ResetDeviceTensor(DataType::TYPE_FP16, {1, seq_len, hidden_units * 3});

  Tensor input_kv_list = input_tensors[2];
  void* key_cache = input_kv_list.GetPtr<void>();
  void* val_cache = key_cache + input_kv_list.GetTotalBytes() / 2;

  std::vector<void*> tmp_buffers;
  tmp_buffers.push_back(input_tensors[8].GetPtr<void>());
  tmp_buffers.push_back(input_tensors[9].GetPtr<void>());
  tmp_buffers.push_back(input_tensors[10].GetPtr<void>());
  tmp_buffers.push_back(input_tensors[11].GetPtr<void>());
  tmp_buffers.push_back(input_tensors[12].GetPtr<void>());

  aclTensor* output;
  this->ascend_flash_attn_->Forward(input_tensor, token_pos, &key_cache, &val_cache, tmp_buffers, &output, false,
                                    context_->GetComputeStreams()[rank_].Get(), GetWorkSpaceFunc());

  size_t size = seq_len * hidden_units;
  ACL_CHECK(aclrtMemcpy(output_tensors[0].GetPtr<void>(), size, tmp_buffers[1], size,
                        aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE));

  output_tensors[0].shape = {static_cast<unsigned long>(seq_len), static_cast<unsigned long>(hidden_units)};
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}
template class PagedAttentionLayer<float>;
template class PagedAttentionLayer<float16>;

}  // namespace ksana_llm
