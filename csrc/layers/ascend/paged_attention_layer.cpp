/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/paged_attention_layer.h"
#include "kernels/ascend/kernel_wrapper.h"

#include "csrc/utils/ascend/common.h"

namespace ksana_llm {

Status PagedAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                 std::shared_ptr<Context> context, int rank) {
  AttentionLayer::Init(parameters, runtime_config, context, rank);
  if (atb_paged_attn_ == nullptr) {
    atb_paged_attn_ = std::make_shared<llm_kernels::ascend::ATBAttention>();
    atb_paged_attn_->Initialize(static_cast<aclDataType>(this->inter_data_type_),
                                static_cast<uint32_t>(this->max_batch_size_), this->num_heads_, this->num_kv_heads_,
                                this->head_size_, this->layer_num_, this->layer_index_, this->block_token_num_,
                                context->GetComputeStreams()[rank].Get(), rank, /*is_multi_token_forward*/ false,
                                this->max_position_embeddings_, this->base_);
  }
  return Status();
}

Status PagedAttentionLayer::Forward(const PagedAttentionInputs& inputs, std::vector<Tensor>& output_tensors) {
  void* output = output_tensors[0].GetPtr<void>();
  void* qkv_tensor = inputs.qkv_tensor->GetPtr<void>();
  int64_t hidden_units = this->num_heads_ * this->head_size_;
  int64_t batch_size = inputs.forward_shape->shape[0];
  output_tensors[0].shape = {static_cast<uint64_t>(batch_size), static_cast<uint64_t>(hidden_units)};
  output_tensors[0].dtype = inputs.qkv_tensor->dtype;
  reinterpret_cast<atb::Context*>(GetRuntimeContext())
      ->SetExecuteStream(this->context_->GetComputeStreams()[this->rank_].Get());
  std::vector<int32_t> seq_len_host(batch_size);
  void* rotary_embedding_pos = inputs.rotary_embedding_pos->GetPtr<void>();
  void* k_cache = inputs.k_cache_blocks_base->GetPtr<void>();
  void* v_cache = inputs.v_cache_blocks_base->GetPtr<void>();
  int32_t* seq_len_host_ptr = inputs.seq_len_host->GetPtr<int32_t>();
  std::memcpy(seq_len_host.data(), seq_len_host_ptr, batch_size * sizeof(int32_t));
  uint64_t* atb_attention_attr_ptr = inputs.atb_attention_attr->GetPtr<uint64_t>();
  uint64_t slot_mapping_dim_1 = atb_attention_attr_ptr[0];
  uint64_t max_num_blocks_per_query = atb_attention_attr_ptr[1];
  int32_t total_block_num = inputs.k_cache_blocks_base->shape[0];
  int32_t total_token_num = inputs.qkv_tensor->shape[0];
  int32_t* slot_mapping = inputs.layers_slot_mapping->GetPtr<int32_t>() + this->layer_index_ * slot_mapping_dim_1;
  int32_t* blocks_table =
      inputs.layers_block_table->GetPtr<int32_t>() + this->layer_index_ * batch_size * max_num_blocks_per_query;
  atb_paged_attn_->Forward(
      output, qkv_tensor, rotary_embedding_pos, reinterpret_cast<void*>(slot_mapping), k_cache, v_cache,
      /*block_tables*/ reinterpret_cast<void*>(blocks_table),
      /*max_num_blocks_per_query*/ max_num_blocks_per_query, static_cast<uint32_t>(batch_size),
      static_cast<uint32_t>(total_token_num), static_cast<uint32_t>(total_block_num), this->block_token_num_,
      static_cast<uint32_t>(this->layer_index_), seq_len_host.data(),
      /*is_multi_token_forward*/ false, reinterpret_cast<atb::Context*>(GetRuntimeContext()), GetWorkSpaceFunc());
  return Status();
}

}  // namespace ksana_llm
