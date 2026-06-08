/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/flash_attention_layer.h"
#include <cstdlib>
#include "kernels/ascend/kernel_wrapper.h"

#include "csrc/utils/ascend/common.h"

namespace ksana_llm {

Status FlashAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                 std::shared_ptr<Context> context, int rank) {
  AttentionLayer::Init(parameters, runtime_config, context, rank);
  if (atb_flash_attn_ == nullptr) {
    atb_flash_attn_ = std::make_shared<llm_kernels::ascend::ATBAttention>();
    atb_flash_attn_->Initialize(static_cast<aclDataType>(this->inter_data_type_),
                                static_cast<uint32_t>(this->max_batch_size_), this->num_heads_, this->num_kv_heads_,
                                this->head_size_, this->layer_num_, this->layer_index_, this->block_token_num_,
                                context->GetComputeStreams()[rank].Get(), rank,
                                /*is_multi_token_forward*/ true, this->max_position_embeddings_, this->base_);

    KLLM_LOG_DEBUG << "FlashAttentionLayer Init, layer_num:" << this->layer_num_;
  }
  return Status();
}

Status FlashAttentionLayer::Forward(const AclForwardParams& p, Tensor* output, const LayerExecContext& exec_ctx) {
  output->shape[0] = p.qkv->shape[0];
  output->shape[1] = this->num_heads_ * this->head_size_;
  output->dtype = p.qkv->dtype;

  void* qkv_tensor = p.qkv->GetPtr<void>();
  int32_t total_token_num = static_cast<int32_t>(p.qkv->shape[0]);
  reinterpret_cast<atb::Context*>(GetRuntimeContext())
      ->SetExecuteStream(this->context_->GetComputeStreams()[this->rank_].Get());
  void* k_cache = p.k_cache_blocks_base->GetPtr<void>();
  void* v_cache = p.v_cache_blocks_base->GetPtr<void>();
  int32_t* seq_len_host_ptr = p.seq_len_host->GetPtr<int32_t>();
  uint64_t* atb_attention_attr_ptr = p.atb_attention_attr->GetPtr<uint64_t>();
  uint64_t slot_mapping_dim_1 = atb_attention_attr_ptr[0];
  int32_t total_block_num = static_cast<int32_t>(p.k_cache_blocks_base->shape[0]);
  int32_t* slot_mapping = p.layers_slot_mapping->GetPtr<int32_t>() + this->layer_index_ * slot_mapping_dim_1;
  void* rotary_embedding_pos = p.rotary_embedding_pos->GetPtr<void>();

  atb_flash_attn_->Forward(
      output->GetPtr<void>(), qkv_tensor, rotary_embedding_pos, reinterpret_cast<void*>(slot_mapping), k_cache, v_cache,
      /*block_tables*/ nullptr,
      /*max_num_blocks_per_query*/ 0, static_cast<uint32_t>(p.batch_size), static_cast<uint32_t>(total_token_num),
      static_cast<uint32_t>(total_block_num), this->block_token_num_, static_cast<uint32_t>(this->layer_index_),
      seq_len_host_ptr,
      /*is_multi_token_forward*/ true, reinterpret_cast<atb::Context*>(GetRuntimeContext()), GetWorkSpaceFunc());
  return Status();
}

}  // namespace ksana_llm
