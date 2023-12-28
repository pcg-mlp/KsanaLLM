/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/paged_attention_layer.h"
#include "numerous_llm/kernels/nvidia/kernel_wrapper.h"


namespace numerous_llm {
Status PagedAttentionLayer::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  AttentionLayer::Init(parameters, context, rank);
  // rotary_embedding_layer_.Init({max_position_embeddings_}, context_, rank_);
  return Status();
}

/*
kv_list  [layers_num * (total_blocks * 2)]
|              layer1               |
| bs1 |     bs2   | bs1 |     bs2   |
|k|k|k|k|k|k|k|k|k|v|v|v|v|v|v|v|v|v|
每个k,v代表一个指针,存储的数据个数为一个block块能存的token个数
需要在model中将block按kv分开存储指针，方便后续计算
*/
Status PagedAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(); 
  // TODO: rotary_embedding_layer 未完成
  // PagedAttention部分
  const Tensor& query = input_tensors[0];
  const Tensor& context_lens = input_tensors[1];
  // 块的位移情况
  // 如上kv_list的情况
  // 一块有8个token时
  // context_lens是0,17,58
  // cache_offset是0,3,9
  const Tensor& cache_offset = input_tensors[2];
  Tensor& out = output_tensors[0];
  Tensor& kv_list = output_tensors[1];
  Tensor& workspace = output_tensors[2];
  int layers_num = output_tensors[1].shape[0];
  int total_blocks = output_tensors[1].shape[1] / 2;
  void** key_cache_ptrs = reinterpret_cast<void**>(layer_index_ * total_blocks * 2);
  void** value_cache_ptrs = reinterpret_cast<void**>(layer_index_ * total_blocks * 2 + total_blocks);
  int max_tokens; //TODO
  int num_seqs; //TODO
  int num_heads; //TODO
  int head_size; //TODO
  int num_kv_heads; //TODO
  int block_size; //TODO  这里不字节数，是存储的token个数
  run_paged_attention<half>(out.GetPtr<void>(), query.GetPtr<void>(),key_cache_ptrs, value_cache_ptrs, context_lens.GetPtr<void>(), max_tokens, context_->GetComputeStreams()[rank_], cache_offset.GetPtr<void>(), 
     num_seqs,
     num_heads,
     head_size,
     num_kv_heads,
     block_size, workspace.GetPtr<void>(), workspace.GetTotalBytes() , {} );
  return Status();
}

}  // namespace numerous_llm
