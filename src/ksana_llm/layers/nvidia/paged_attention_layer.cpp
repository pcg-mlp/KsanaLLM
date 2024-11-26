/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status PagedAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Init(const std::vector<std::any>& parameters,
                                                              std::shared_ptr<Context> context, int rank) {
  return AttentionLayer<SCALAR_T>::Init(parameters, context, rank);
}

/*
kv_list  [layers_num * (total_blocks * 2)]
|              layer1               |
| bs1 |     bs2   | bs1 |     bs2   |
|k|k|k|k|k|k|k|k|k|v|v|v|v|v|v|v|v|v|
每个k,v代表一个指针,存储的数据个数为一个block块能存的token个数
需要在model中将block按kv分开存储指针，方便后续计算
*/
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status PagedAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Forward(const std::vector<Tensor>& input_tensors,
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
  //   7: forward_shape (context_num, context_max_tokens, context_total_block_num, decode_num, decode_max_tokens,
  //      decode_total_block_num)
  //   8: 用于存储 qk 的临时空间(TODO:)
#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
  //   9: kv_cache_base_ptr_tensor: [layer_num, 2]
  //   10: block_table: [batch_size, max_tokens]
#endif
  // output_tensors:
  //   0: paged attention output
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
  size_t context_num = input_tensors[7].shape[0];
  const Tensor& rotary_embedding_pos = input_tensors[4];
  const Tensor& rotary_embedding_mask = input_tensors[5];
  const Tensor& workspace = input_tensors[6];
  const Tensor& qkv_workspace = input_tensors[8];
  int layer_block_num = input_tensors[7].shape[5];
  int context_layer_block_num = input_tensors[7].shape[2];
  int max_tokens = input_tensors[7].shape[4];
  int batch_size = input_tensors[7].shape[3];
  int total_tokens = batch_size;
  size_t context_tokens = query.shape[0] - batch_size;
  void** k_list_base = (kv_list.GetPtr<void*>() + this->layer_num_ * context_layer_block_num * 2);
  void** k_list = k_list_base + static_cast<size_t>(this->layer_index_ * layer_block_num * 2);
  void** v_list = k_list + layer_block_num;
  Tensor& out = output_tensors[0];
  out.dtype = query.dtype;
  out.shape = {query.shape[0], this->num_heads_ * (size_t)this->head_size_};

#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
  int64_t kv_cache_block_num = *(input_tensors[9].GetPtr<int64_t>());
  void** layer_kv_cache_ptr = input_tensors[9].GetPtr<void*>() + 1;
  void* k_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2];
  void* v_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2 + 1];
  int32_t* block_table_ptr = input_tensors[10].GetPtr<int32_t>();
  int max_blocks_per_seq = input_tensors[10].shape[1];
#endif

  auto skipped_context_out_ptr = out.GetPtr<void>() + context_tokens * (out.GetTotalBytes() / out.shape[0]);
  auto skipped_context_query_ptr = query.GetPtr<void>() + context_tokens * (query.GetTotalBytes() / query.shape[0]);
  auto skipped_context_cache_offset_ptr =
      cache_offset.GetPtr<void>() + (context_num + 1) * GetTypeSize(cache_offset.dtype);
  auto skipped_context_rotary_embedding_pos_ptr =
      rotary_embedding_pos.GetPtr<void>() +
      (rotary_embedding_pos.shape[0] - batch_size) * GetTypeSize(rotary_embedding_pos.dtype);
  auto skipped_context_rotary_embedding_mask_ptr =
      rotary_embedding_mask.GetPtr<void>() +
      (rotary_embedding_mask.shape[0] - batch_size) * GetTypeSize(rotary_embedding_mask.dtype);

  InvokePagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(
      skipped_context_out_ptr, skipped_context_query_ptr, k_list, v_list, context_lens.GetPtr<void>(), max_tokens,
      this->context_->GetComputeStreams()[this->rank_].Get(), skipped_context_cache_offset_ptr, batch_size,
      this->num_heads_, this->head_size_, this->num_kv_heads_, this->stride_size_, this->block_token_num_,
      this->k_scale_, this->v_scale_, batch_size, skipped_context_rotary_embedding_pos_ptr,
      skipped_context_rotary_embedding_mask_ptr, total_tokens, this->rotary_embedding_cuda_, workspace.GetPtr<void>(),
#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
      workspace.GetTotalBytes(), this->rank_, this->alibi_slopes_, qkv_workspace.GetPtr<void>(), k_cache_ptr,
      v_cache_ptr, block_table_ptr, kv_cache_block_num, max_blocks_per_seq);
#else
      workspace.GetTotalBytes(), this->rank_, this->alibi_slopes_, qkv_workspace.GetPtr<void>());
#endif

  return Status();
}

using llm_kernels::utils::KVCacheType;
template class PagedAttentionLayer<float, float, KVCacheType::kAuto>;
template class PagedAttentionLayer<float, uint8_t, KVCacheType::kFp8E4M3>;
template class PagedAttentionLayer<float, uint8_t, KVCacheType::kFp8E5M2>;
template class PagedAttentionLayer<half, half, KVCacheType::kAuto>;
template class PagedAttentionLayer<half, uint8_t, KVCacheType::kFp8E4M3>;
template class PagedAttentionLayer<half, uint8_t, KVCacheType::kFp8E5M2>;
#ifdef ENABLE_BFLOAT16
template class PagedAttentionLayer<__nv_bfloat16, __nv_bfloat16, KVCacheType::kAuto>;
template class PagedAttentionLayer<__nv_bfloat16, uint8_t, KVCacheType::kFp8E4M3>;
template class PagedAttentionLayer<__nv_bfloat16, uint8_t, KVCacheType::kFp8E5M2>;
#endif

}  // namespace ksana_llm
