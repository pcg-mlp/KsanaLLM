/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/paged_attention_layer.h"
#include "numerous_llm/kernels/nvidia/kernel_wrapper.h"

namespace numerous_llm {

/*
kv_list  [layers_num * (total_blocks * 2)]
|              layer1               |
| bs1 |     bs2   | bs1 |     bs2   |
|k|k|k|k|k|k|k|k|k|v|v|v|v|v|v|v|v|v|
每个k,v代表一个指针,存储的数据个数为一个block块能存的token个数
需要在model中将block按kv分开存储指针，方便后续计算
*/
Status PagedAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // PagedAttention部分
  // input_tensors:
  //   0: 输入数据
  //   1: input_offset_tensor (uint64 版本)
  //   2: kv_cache_offset_tensor
  //   3: forward_shape
  // output_tensors:
  //   0: paged attention output
  //   1: kv_list
  //   2: workspace 空间(TODO: 计算大小)
  const Tensor& query = input_tensors[0];
  const Tensor& context_lens = input_tensors[1];
  // 块的位移情况
  // 如上kv_list的情况
  // 一块有8个token时
  // context_lens是0,17,58
  // cache_offset是0,3,9
  const Tensor& cache_offset = input_tensors[2];
  const Tensor& forward_shape = input_tensors[3];
  int batch_size = forward_shape.shape[0];
  int max_tokens = forward_shape.shape[1];

  Tensor& out = output_tensors[0];
  Tensor& kv_list = output_tensors[1];
  Tensor& workspace = output_tensors[2];
  int layers_num = kv_list.shape[0];
  int total_blocks = kv_list.shape[1] / 2;
  void** key_cache_ptrs = reinterpret_cast<void**>(layer_index_ * total_blocks * 2);
  void** value_cache_ptrs = reinterpret_cast<void**>(layer_index_ * total_blocks * 2 + total_blocks);

  NLLM_LOG_INFO  <<  fmt::format("batch_size = {}, total_blocks = {}, layers_num = {}, max_tokens = {}", batch_size,
                                 total_blocks, layers_num, max_tokens);
  run_paged_attention<half>(out.GetPtr<void>(), query.GetPtr<void>(), key_cache_ptrs, value_cache_ptrs,
                            context_lens.GetPtr<void>(), max_tokens, context_->GetComputeStreams()[rank_],
                            cache_offset.GetPtr<void>(), batch_size, num_heads_, head_size_, num_kv_heads_, block_size_,
                            workspace.GetPtr<void>(), workspace.GetTotalBytes(), {});
  return Status();
}

}  // namespace numerous_llm
