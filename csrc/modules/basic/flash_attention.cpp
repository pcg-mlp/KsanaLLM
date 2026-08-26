/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/flash_attention.h"
#include <algorithm>
#include "layers/flash_attention_layer.h"
#include "models/base/attention_meta/attention_meta_builder.h"
#include "utils/singleton.h"

namespace ksana_llm {

FlashAttention::FlashAttention(bool is_neox, const LayerCreationContext& creation_context,
                               const AttentionCreationConfig& attn_config)
    : reuse_prefix_caching_(attn_config.reuse_prefix_caching),
      context_(creation_context.context),
      rank_(creation_context.rank),
      enable_blocked_multi_token_forwarding_kv_(
          creation_context.runtime_config.attn_backend_config.enable_blocked_multi_token_forwarding_kv) {
  uint32_t zero = 0;
  flash_attention_layer_ = std::make_shared<FlashAttentionLayer>();

  std::vector<std::any> attention_param;
  attention_param.push_back(attn_config.model_config.quant_config.method);  // for quant method
  attention_param.push_back(attn_config.model_config.layernorm_eps);        // for q k layernorm
  attention_param.push_back(attn_config.model_config.use_qk_norm);
  attention_param.push_back(attn_config.idx);
  attention_param.push_back(attn_config.layer_num_on_node);
  attention_param.push_back(attn_config.max_position_embeddings);
  attention_param.push_back(attn_config.head_num_per_tp);
  attention_param.push_back(attn_config.num_kv_heads_per_tp);
  attention_param.push_back(attn_config.size_per_head);
  attention_param.push_back(attn_config.stride_size);
  attention_param.push_back(attn_config.tensor_para_size);
  attention_param.push_back(attn_config.kv_cache_dtype);
  attention_param.push_back(
      attn_config.model_config.k_scales[attn_config.idx + creation_context.pipeline_config.lower_layer_idx]);
  attention_param.push_back(
      attn_config.model_config.v_scales[attn_config.idx + creation_context.pipeline_config.lower_layer_idx]);
  attention_param.push_back(attn_config.rotary_embedding);
  attention_param.push_back(attn_config.rope_theta);
  // new add for mla
  attention_param.push_back(zero);
  attention_param.push_back(zero);
  attention_param.push_back(zero);
  attention_param.push_back(zero);
  attention_param.push_back(zero);
  // end new add for mla
  attention_param.push_back(is_neox);
  attention_param.push_back(attn_config.position_encoding);
  attention_param.push_back(attn_config.cos_sin_cache_ptr);
  attention_param.push_back(attn_config.model_config.rope_scaling_factor_config);
  attention_param.push_back(attn_config.max_batch_size);
  // add for applying temperature tuning
  attention_param.push_back(attn_config.model_config.attn_temperature_tuning);
  attention_param.push_back(attn_config.model_config.attn_scale);
  attention_param.push_back(attn_config.model_config.floor_scale);
  // end for applying temperature tuning
  std::vector<std::any> flash_attention_param = attention_param;
  flash_attention_param.push_back(true);
  if (attn_config.model_config.rope_scaling_factor_config.type == "mrope") {
    flash_attention_param.push_back(attn_config.mrope_section_ptr);
    use_mrotary_ = true;
  } else if (attn_config.model_config.rope_scaling_factor_config.type == "xdrope") {
    flash_attention_param.push_back(attn_config.xdrope_section_ptr);
    use_xdrotary_ = true;
  } else {
    flash_attention_param.push_back(nullptr);
  }
  flash_attention_param.push_back(attn_config.model_config.enable_qk_pre_norm_before_rotary_pos);
  flash_attention_layer_->Init(flash_attention_param, creation_context.runtime_config, context_, rank_);
  // Workspace sizing for Kunlun: Q + K + V contiguous buffers.
  // Must be sized by max_step_token_num (total tokens in one forward step), NOT max_batch_size
  // (number of concurrent requests).  Using max_batch_size causes workspace overflow when
  // max_token_len or max_step_tokens is increased, leading to Q/K/V buffer overrun and garbled output.
  //
  // Element size must track the model's actual dtype: the workspace is written as T*
  // (T = SCALAR_T from DISPATCH_BY_DTYPE_AND_KVTYPE) inside ForwardT.  Hard-coding
  // sizeof(uint16_t) undersizes the buffer for fp32 and would silently overrun.
  size_t flash_ws_size;
#ifdef ENABLE_KUNLUN
  flash_ws_size = static_cast<size_t>(creation_context.runtime_config.max_step_token_num) *
                  static_cast<size_t>(attn_config.stride_size) * GetTypeSize(attn_config.model_config.weight_data_type);
#else
  flash_ws_size =
      static_cast<size_t>(attn_config.max_batch_size) * static_cast<size_t>(attn_config.stride_size) * sizeof(uint16_t);
#endif
  // Ensure at least 8MB for safety (covers max_step_tokens * stride * fp16).
  flash_ws_size = std::max(flash_ws_size, static_cast<size_t>(8 * 1024 * 1024));
  flash_attention_layer_->SetWorkspaceBuffer(creation_context.workspace_mgr->GetWorkspace(flash_ws_size));
}

FlashAttention::~FlashAttention() {}

Status FlashAttention::Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::shared_ptr<ModelInput>& model_input,
                               std::vector<Tensor>& hidden_buffer_tensors_1, std::vector<Tensor>& shared_buffer_tensors,
                               const AttentionForwardContext& forward_context, Tensor query_layernorm_weight,
                               Tensor key_layernorm_weight) {
  if (reuse_prefix_caching_ && !enable_blocked_multi_token_forwarding_kv_) {
    AddAttentionPrefixCache(hidden_buffer_tensors_0, model_input, hidden_buffer_tensors_1, shared_buffer_tensors);
  }

#ifdef ENABLE_CUDA
  auto* typed_flash_layer = static_cast<FlashAttentionLayer*>(flash_attention_layer_.get());
  const auto& fs = forward_context.forward_shape;
  const Tensor& rotary_pos = use_mrotary_ ? model_input->dp_mrotary_embedding_pos
                                          : (use_xdrotary_ ? model_input->dp_xdrotary_embedding_pos
                                                           : model_input->flash_meta->rotary_embedding_pos);
  const auto* std_meta = model_input->shared_attention_meta->GetStandard();
  FlashAttentionLayer::CudaForwardParams params;
  params.qkv = &hidden_buffer_tensors_0[0];
  params.token_offset = &model_input->dp_input_offset_uint64_tensor;
  params.kv_list = &model_input->flash_meta->kv_list;
  params.prefix_offset = &model_input->dp_input_prefix_uint64_tensor;
  params.kv_cache_offset = &model_input->flash_meta->kv_cache_offset;
  params.rotary_embedding_pos = &rotary_pos;
  params.rotary_embedding_mask = &model_input->flash_meta->rotary_embedding_mask;
  params.src_flexible_rotary_embedding_pos = &std_meta->dp_src_flexible_rotary_embedding_pos;
  params.flexible_rotary_embedding_mask = &std_meta->dp_flexible_rotary_embedding_mask;
  params.dst_flexible_kv_cache = &std_meta->dp_dst_flexible_kv_cache_tensor;
  params.src_flexible_kv_cache = &std_meta->dp_src_flexible_kv_cache_tensor;
  params.dst_flexible_token_idx = &std_meta->dp_dst_flexible_token_idx_tensor;
  params.src_flexible_token_idx = &std_meta->dp_src_flexible_token_idx_tensor;
  params.flexible_offset_uint64 = &std_meta->dp_flexible_offset_uint64_tensor;
  params.layer_block_num = static_cast<int>(fs.shape[2]);
  // shape[12] (dp_decode_tokens) is the decode-stage token count; shape[3] is the decode request
  // count. Under MTP each decode request carries more than one token, and shape[12] is also the
  // dp-scoped value that matches qkv->shape[0].
  params.decode_token_num = static_cast<int>(fs.shape[12]);
  params.max_forwarding_tokens = static_cast<int>(fs.shape[6]);
  params.batch_size = static_cast<int>(fs.shape[7]);
  params.max_tokens = static_cast<int>(fs.shape[8]);
  params.query_layernorm_weight = &query_layernorm_weight;
  params.key_layernorm_weight = &key_layernorm_weight;
  params.use_cache = forward_context.flag_tensor.GetPtr<bool>()[0];
  params.kv_cache_base_ptr = &std_meta->layer_kv_cache_ptr;
  params.block_table = &model_input->flash_meta->block_table;
  params.input_without_prefix_offset = &model_input->dp_prefill_q_offset_uint64_tensor;
  STATUS_CHECK_RETURN(typed_flash_layer->Forward(params, &hidden_buffer_tensors_1[0]));
#elif defined(ENABLE_ACL)
  auto* typed_flash_layer = static_cast<FlashAttentionLayer*>(flash_attention_layer_.get());
  const auto& fs = forward_context.forward_shape;
  FlashAttentionLayer::AclForwardParams params;
  params.qkv = &hidden_buffer_tensors_0[0];
  params.rotary_embedding_pos = &model_input->flash_meta->rotary_embedding_pos;
  params.layers_slot_mapping = &model_input->layers_slot_mapping;
  params.k_cache_blocks_base = &model_input->k_cache_blocks_base;
  params.v_cache_blocks_base = &model_input->v_cache_blocks_base;
  params.seq_len_host = &model_input->seq_len_host;
  params.batch_size = static_cast<int>(fs.shape[0]);
  params.max_tokens = static_cast<int>(fs.shape[1]);
  params.layer_block_num = static_cast<int>(fs.shape[2]);
  params.atb_attention_attr = &model_input->atb_attention_attr;
  STATUS_CHECK_RETURN(typed_flash_layer->Forward(params, &hidden_buffer_tensors_1[0]));
#elif defined(ENABLE_KUNLUN)
  // inference on Kunlun XPU — same tensor protocol as Ascend/ATB + QK Norm weights
  auto* typed_flash_layer = static_cast<FlashAttentionLayer*>(flash_attention_layer_.get());
  const auto& fs = forward_context.forward_shape;
  FlashAttentionLayer::KunlunForwardParams params;
  params.qkv = &hidden_buffer_tensors_0[0];
  params.rotary_embedding_pos = &model_input->flash_meta->rotary_embedding_pos;
  params.layers_slot_mapping = &model_input->layers_slot_mapping;
  params.k_cache_blocks_base = &model_input->k_cache_blocks_base;
  params.v_cache_blocks_base = &model_input->v_cache_blocks_base;
  params.seq_len_host = &model_input->seq_len_host;
  // Kunlun uses the CUDA-style 13-D forward_shape layout (see forwarding_context.cpp), NOT the ACL
  // 3-D layout: shape[0] is the GLOBAL multi_token_request_num while the flash layer needs the
  // DP-scoped batch. shape[7]=dp_multi_token_request_num (batch_size, = len(qlod)-1) and
  // shape[8]=dp_multi_token_request_max_tokens. The ACL indices [0]/[1] were copied here by
  // mistake; when attention DP>1, shape[0]>shape[7] so max_seq_len over-reads seq_len_host and
  // batch_size desyncs from qlod → NaN prefill attention. Matches the CUDA branch and reference.
  params.batch_size = static_cast<int>(fs.shape[7]);
  params.max_tokens = static_cast<int>(fs.shape[8]);
  params.layer_block_num = static_cast<int>(fs.shape[2]);
  params.atb_attention_attr = &model_input->atb_attention_attr;
  params.query_layernorm_weight = &query_layernorm_weight;
  params.key_layernorm_weight = &key_layernorm_weight;
  params.qlod = &model_input->dp_prefill_q_offset_int32_host_tensor;
  STATUS_CHECK_RETURN(typed_flash_layer->Forward(params, &hidden_buffer_tensors_1[0]));
#endif
  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);

  if (reuse_prefix_caching_ && !enable_blocked_multi_token_forwarding_kv_) {
    RemoveAttentionPrefixCache(hidden_buffer_tensors_0, model_input, hidden_buffer_tensors_1, shared_buffer_tensors);
  }

  return Status();
}

Status FlashAttention::AddAttentionPrefixCache(std::vector<Tensor>& hidden_buffer_tensors_0,
                                               std::shared_ptr<ModelInput>& model_input,
                                               std::vector<Tensor>& hidden_buffer_tensors_1,
                                               std::vector<Tensor>& shared_buffer_tensors) {
  // The input shape for Flash Attention must match the actual token_num, which includes the lengths of both prefix and
  // speculative tokens
  // The mmha_prefix_input reserves space for all tokens. Here, the tokens from mmha_origin_input are copied to their
  // corresponding positions in mmha_prefix_input. For the remaining tokens in mmha_prefix_input, their key-value (KV)
  // pairs will be copied from the KV cache during Flash Attention, while the query (Q) can remain unused.
  const auto& mmha_origin_input = hidden_buffer_tensors_0[0];  // [token_num_without_prefix, hidden]
  auto& mmha_prefix_input = hidden_buffer_tensors_1[0];        // [token_num_with_prefix, hidden]
  size_t total_token_num = 0;                                  // 包含prefix、speculate的实际token总数
  const size_t dtype_size = GetTypeSize(mmha_origin_input.dtype);
  const size_t size_per_token = mmha_origin_input.shape[1] * dtype_size;
  for (size_t idx = 0; idx < model_input->dp_batch_size; ++idx) {
    // The single token is located in the latter half; copy it all at once and then exit.
    // This check must precede the prefix_length calculation (dp_input_prefix_list_uint64[idx+1])
    // because the prefix list only contains entries for multi-token requests.
    if (idx >= model_input->dp_multi_token_request_num && model_input->dp_single_token_request_num > 0) {
      const size_t input_offset = model_input->dp_input_offset_uint64_host.GetPtr<size_t>()[idx];
      const size_t prefix_offset = model_input->dp_input_prefix_list_uint64[idx];
      const size_t src_offset = (input_offset - prefix_offset) * size_per_token;
      const size_t remaining_bytes = mmha_origin_input.GetTotalBytes() - src_offset;
      MemcpyAsync(shared_buffer_tensors[0].template GetPtr<void>(),
                  mmha_origin_input.template GetPtr<void>() + src_offset, remaining_bytes, MEMCPY_DEVICE_TO_DEVICE,
                  context_->device->GetPrimaryComputeStream());
      shared_buffer_tensors[0].shape = {remaining_bytes / dtype_size / model_input->dp_single_token_request_num,
                                        model_input->dp_single_token_request_num};
      total_token_num += model_input->dp_single_token_request_num;
      break;
    }
    const size_t input_offset = model_input->dp_input_offset_uint64_host.GetPtr<size_t>()[idx];
    const size_t prefix_offset = model_input->dp_input_prefix_list_uint64[idx];
    const size_t src_offset = (input_offset - prefix_offset) * size_per_token;
    const size_t input_length = model_input->dp_input_offset_uint64_host.GetPtr<size_t>()[idx + 1] - input_offset;
    const size_t prefix_length = model_input->dp_input_prefix_list_uint64[idx + 1] - prefix_offset;
    const size_t copy_size = (input_length - prefix_length) * size_per_token;
    const size_t dst_offset = (input_offset + prefix_length) * size_per_token;
    // Copy multi tokens
    MemcpyAsync(mmha_prefix_input.template GetPtr<void>() + dst_offset,
                mmha_origin_input.template GetPtr<void>() + src_offset, copy_size, MEMCPY_DEVICE_TO_DEVICE,
                context_->device->GetPrimaryComputeStream());
    total_token_num += input_length;
  }
  mmha_prefix_input.shape = {total_token_num, mmha_origin_input.shape[1]};
  mmha_prefix_input.dtype = mmha_origin_input.dtype;
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  return Status();
}

Status FlashAttention::RemoveAttentionPrefixCache(std::vector<Tensor>& hidden_buffer_tensors_0,
                                                  std::shared_ptr<ModelInput>& model_input,
                                                  std::vector<Tensor>& hidden_buffer_tensors_1,
                                                  std::vector<Tensor>& shared_buffer_tensors) {
  // After the completion of MMHA inference, copy the data from the MMHA output results,
  // excluding the Prefix Cache section, and continue with the subsequent inference.
  auto& mmha_prefix_output = hidden_buffer_tensors_0[0];
  auto& mmha_output = hidden_buffer_tensors_1[0];
  auto attention_input_shape = hidden_buffer_tensors_1[0].shape;
  size_t total_token_num_without_prefix = 0;
  size_t dst_offset = 0;
  size_t src_offset = 0;
  size_t dtype_size = GetTypeSize(mmha_prefix_output.dtype);
  size_t size_per_token = mmha_prefix_output.shape[1] * dtype_size;
  for (size_t idx = 0; idx < model_input->dp_multi_token_request_num; ++idx) {
    size_t prefix_length =
        model_input->dp_input_prefix_list_uint64[idx + 1] - model_input->dp_input_prefix_list_uint64[idx];
    size_t input_length = model_input->dp_input_offset_uint64_host.GetPtr<size_t>()[idx + 1] -
                          model_input->dp_input_offset_uint64_host.GetPtr<size_t>()[idx];
    src_offset += prefix_length * size_per_token;
    size_t copy_size = size_per_token * (input_length - prefix_length);

    MemcpyAsync(mmha_output.template GetPtr<void>() + dst_offset,
                mmha_prefix_output.template GetPtr<void>() + src_offset, copy_size, MEMCPY_DEVICE_TO_DEVICE,
                context_->device->GetPrimaryComputeStream());
    src_offset += copy_size;
    dst_offset += copy_size;
    total_token_num_without_prefix += input_length - prefix_length;
  }
  mmha_output.shape = {total_token_num_without_prefix, mmha_prefix_output.shape[1]};
  mmha_output.dtype = mmha_prefix_output.dtype;
  if (model_input->dp_single_token_request_num > 0) {
    MemcpyAsync(hidden_buffer_tensors_0[0].template GetPtr<void>() + shared_buffer_tensors[0].GetTotalBytes() /
                                                                         model_input->dp_single_token_request_num *
                                                                         total_token_num_without_prefix,
                shared_buffer_tensors[0].template GetPtr<void>(), shared_buffer_tensors[0].GetTotalBytes(),
                MEMCPY_DEVICE_TO_DEVICE, context_->device->GetPrimaryComputeStream());
    hidden_buffer_tensors_0[0].shape = {total_token_num_without_prefix + model_input->dp_single_token_request_num,
                                        attention_input_shape[1]};
  }
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  return Status();
}

}  // namespace ksana_llm
