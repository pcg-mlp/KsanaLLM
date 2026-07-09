/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/flash_attention_layer.h"
#include "executor/layer_progress_tracker.h"
#include "kernels/nvidia/kernel_wrapper.h"
#include "pd_v2/prefill/pd_v2_layer_event_tracker.h"
#include "utils/singleton.h"
#include "utils/string_utils.h"

namespace ksana_llm {

Status FlashAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                 std::shared_ptr<Context> context, int rank) {
  enable_blocked_multi_token_forwarding_kv_ =
      runtime_config.attn_backend_config.enable_blocked_multi_token_forwarding_kv;
  use_flashinfer_for_decode_ = runtime_config.attn_backend_config.use_flashinfer_for_decode;
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashAttentionLayer::ForwardT(const CudaForwardParams& p, Tensor* output) {
  int max_tokens = p.max_tokens;
  int batch_size = p.batch_size;
  int layer_block_num = p.layer_block_num;
  int total_tokens = static_cast<int>(p.qkv->shape[0]) - p.single_token_request_num;

  void** k_list = p.kv_list->GetPtr<void*>() + this->layer_index_ * layer_block_num * 2;
  void** v_list = k_list + layer_block_num;

  int64_t kv_cache_block_num = 0;
  void** layer_kv_cache_ptr = nullptr;
  void* k_cache_ptr = nullptr;
  void* v_cache_ptr = nullptr;
  int32_t* block_table_ptr = nullptr;
  int max_blocks_per_seq = 0;
  size_t* input_without_prefix_offset_ptr = nullptr;
  int max_forwarding_tokens = 0;

  if (enable_blocked_multi_token_forwarding_kv_ || use_flashinfer_for_decode_) {
    kv_cache_block_num = *(p.kv_cache_base_ptr->GetPtr<int64_t>());
    layer_kv_cache_ptr = p.kv_cache_base_ptr->GetPtr<void*>() + 1;
    k_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2];
    v_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2 + 1];
    block_table_ptr = p.block_table->GetPtr<int32_t>();
    max_blocks_per_seq = static_cast<int>(p.block_table->shape[1]);
    input_without_prefix_offset_ptr = p.input_without_prefix_offset->GetPtr<size_t>();
    max_forwarding_tokens = p.max_forwarding_tokens;
  }

  AttenVarlen<SCALAR_T, CACHE_T, KV_DTYPE>(
      p.qkv->GetPtr<void>(), p.rotary_embedding_pos->GetPtr<void>(), p.rotary_embedding_mask->GetPtr<void>(),
      output->GetPtr<void>(), p.token_offset->GetPtr<void>(), this->rotary_embedding_cuda_, total_tokens, max_tokens,
      batch_size, this->num_heads_, this->num_kv_heads_, this->head_size_, this->stride_size_, this->k_scale_,
      this->v_scale_, this->tensor_para_size_, this->is_causal_, this->rank_, this->block_token_num_, k_list, v_list,
      p.prefix_offset->GetPtr<void>(), p.kv_cache_offset->GetPtr<void>(), this->alibi_slopes_, this->layer_index_,
      p.src_flexible_rotary_embedding_pos->GetPtr<void>(), p.flexible_rotary_embedding_mask->GetPtr<void>(),
      p.dst_flexible_kv_cache->GetPtr<void>(), p.src_flexible_kv_cache->GetPtr<void>(),
      p.dst_flexible_token_idx->GetPtr<void>(), p.src_flexible_token_idx->GetPtr<void>(),
      p.flexible_offset_uint64->GetPtr<void>(), p.dst_flexible_kv_cache->shape[0], this->layernorm_eps_,
      this->use_qk_norm_, p.query_layernorm_weight->GetPtr<void>(), p.key_layernorm_weight->GetPtr<void>(), p.use_cache,
      this->context_->device->GetPrimaryComputeStream().Get(), k_cache_ptr, v_cache_ptr, block_table_ptr,
      kv_cache_block_num, max_blocks_per_seq, input_without_prefix_offset_ptr, max_forwarding_tokens,
      this->enable_qk_pre_norm_before_rotary_pos_, this->no_rope_, this->attn_temperature_tuning_, this->attn_scale_,
      this->floor_scale_, this->enable_blocked_multi_token_forwarding_kv_, use_flashinfer_for_decode_);

  Singleton<LayerProgressTracker>::GetInstance()->RecordLayerProgress(
      this->rank_, this->layer_index_, this->context_->device->GetPrimaryComputeStream());

  // pd_v2 Prefill: per-layer KV-write hook. The tracker is gated by
  // IsActive() — only fires on Prefill executor processes that have
  // initialized PdV2LayerEventTracker via PdV2KvDispatcher wiring.
  // V1 / Decode / non-pd_v2 paths see IsActive()==false and skip.
  auto pd_v2_tracker = Singleton<pd_v2::PdV2LayerEventTracker>::GetInstance();
  if (pd_v2_tracker->IsActive()) {
    pd_v2_tracker->RecordLayerProgress(this->rank_, this->layer_index_,
                                       this->context_->device->GetPrimaryComputeStream());
  }

  output->shape[0] = static_cast<size_t>(total_tokens);
  output->shape[1] = this->num_heads_ * this->head_size_;
  output->dtype = p.qkv->dtype;
  return Status();
}

Status FlashAttentionLayer::Forward(const CudaForwardParams& p, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_DTYPE_AND_KVTYPE(inter_data_type_, kv_cache_dtype_, ForwardT, p, output);
}

}  // namespace ksana_llm
