/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

/*
kv_list  [layers_num * (total_blocks * 2)]
|              layer1               |
| bs1 |     bs2   | bs1 |     bs2   |
|k|k|k|k|k|k|k|k|k|v|v|v|v|v|v|v|v|v|
每个k,v代表一个指针,存储的数据个数为一个block块能存的token个数
需要在model中将block按kv分开存储指针，方便后续计算
*/
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
  const Tensor& qkv_workspace = input_tensors[7];
  int layer_block_num = input_tensors[6].shape[2];
  int max_tokens = input_tensors[6].shape[1];
  int batch_size = input_tensors[6].shape[0];
  int total_tokens = input_tensors[0].shape[0];
  void** k_list = (kv_list.GetPtr<void*>()) + (size_t)layer_index_ * layer_block_num * 2;
  void** v_list = k_list + layer_block_num;
  Tensor& out = output_tensors[0];
  out.dtype = query.dtype;
  out.shape = {query.shape[0], num_heads_ * (size_t)head_size_};
  InvokePagedAttention<T>(out.GetPtr<void>(), query.GetPtr<void>(), k_list, v_list, context_lens.GetPtr<void>(),
                          max_tokens, context_->GetComputeStreams()[rank_].Get(), cache_offset.GetPtr<void>(),
                          batch_size, num_heads_, head_size_, num_kv_heads_, stride_size_, block_token_num_, batch_size,
                          rotary_embedding_pos.GetPtr<void>(), total_tokens, rotary_embedding_cuda_,
                          workspace.GetPtr<void>(), workspace.GetTotalBytes(), rank_, alibi_slopes_,
                          qkv_workspace.GetPtr<void>());
  return Status();
}

template class PagedAttentionLayer<float>;
template class PagedAttentionLayer<half>;
#ifdef ENABLE_BFLOAT16
template class PagedAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
