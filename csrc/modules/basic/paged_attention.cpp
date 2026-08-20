/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/paged_attention.h"
#include <algorithm>
#include "layers/paged_attention_layer.h"
#include "models/base/attention_meta/attention_meta_builder.h"

namespace ksana_llm {

PagedAttention::PagedAttention(bool is_neox, const LayerCreationContext& creation_context,
                               const AttentionCreationConfig& attn_config) {
  paged_attention_layer_ = std::make_shared<PagedAttentionLayer>();

  uint32_t zero = 0;
  std::vector<std::any> attention_param;
  attention_param.push_back(attn_config.model_config.quant_config.method);
  attention_param.push_back(attn_config.model_config.layernorm_eps);  // for q k layernorm
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
  std::vector<std::any> paged_attention_param = attention_param;
  // NOTE(karlluo): bool for is_multi_token_forward
  paged_attention_param.push_back(false);
  // aligned with flash attention
  paged_attention_param.push_back(nullptr);
  paged_attention_param.push_back(attn_config.model_config.enable_qk_pre_norm_before_rotary_pos);
  paged_attention_layer_->Init(paged_attention_param, creation_context.runtime_config, creation_context.context,
                               creation_context.rank);
  // FlashInfer workspaces will be initialized when SetWorkspaceBuffer is called.
  // Kunlun workspace layout: [Q | K | V | context_len (int32)].
  // context_len array = max_batch * sizeof(int32_t) must be explicitly included, or the
  // MemcpyAsync in paged_attention_layer.cpp writes past the workspace end (OOB write).
  //
  // Element size must track the model's actual dtype (see flash_attention.cpp for the
  // same reasoning): the workspace is written as T* in ForwardT, so hard-coding
  // sizeof(uint16_t) undersizes it for fp32.
#ifdef ENABLE_KUNLUN
  size_t paged_ws_size =
      static_cast<size_t>(attn_config.max_batch_size) *
          (static_cast<size_t>(attn_config.stride_size) *
               GetTypeSize(attn_config.model_config.weight_data_type) +
           sizeof(int32_t));
  paged_ws_size = std::max(paged_ws_size, static_cast<size_t>(8 * 1024 * 1024));
#else
  size_t paged_ws_size = std::max(static_cast<size_t>(8 * 1024 * 1024),
                                  static_cast<size_t>(attn_config.max_batch_size) *
                                  static_cast<size_t>(attn_config.stride_size) * sizeof(uint16_t));
#endif
  paged_attention_layer_->SetWorkspaceBuffer(creation_context.workspace_mgr->GetWorkspace(paged_ws_size));
}

PagedAttention::~PagedAttention() {}

Status PagedAttention::Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::shared_ptr<ModelInput>& model_input,
                               std::vector<Tensor>& hidden_buffer_tensors_1, std::vector<Tensor>& paged_buffer_tensors,
                               Tensor& kv_cache_buffer_tensor, const AttentionForwardContext& forward_context,
                               Tensor query_layernorm_weight, Tensor key_layernorm_weight) {
  // normal page attention only has one input
  // It must be placed at the end of `page_metas`, since they are sorted in descending order by token number
  auto& input_meta = *model_input->page_metas.back();

#ifdef ENABLE_CUDA
  {
    PagedAttentionInputs inputs;
    inputs.query = &hidden_buffer_tensors_0[0];
    inputs.context_lens = &input_meta.input_length;
    inputs.kv_list = &input_meta.kv_list;
    inputs.cache_offset = &input_meta.kv_cache_offset;
    inputs.rotary_embedding_pos = &input_meta.rotary_embedding_pos;
    inputs.rotary_embedding_mask = &input_meta.rotary_embedding_mask;
    inputs.workspace = &kv_cache_buffer_tensor;
    inputs.forward_shape = &forward_context.forward_shape;
    inputs.qkv_workspace = &paged_buffer_tensors[0];
    inputs.query_layernorm_weight = &query_layernorm_weight;
    inputs.key_layernorm_weight = &key_layernorm_weight;
    inputs.layer_kv_cache_ptr = &model_input->shared_attention_meta->GetStandard()->layer_kv_cache_ptr;
    inputs.block_table = &input_meta.block_table;
    STATUS_CHECK_RETURN(paged_attention_layer_->Forward(inputs, hidden_buffer_tensors_1));
  }
  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
#elif defined(ENABLE_ACL)
  {
    PagedAttentionInputs inputs;
    inputs.qkv_tensor = &hidden_buffer_tensors_0[0];
    inputs.rotary_embedding_pos = &input_meta.rotary_embedding_pos;
    inputs.layers_slot_mapping = &model_input->layers_slot_mapping;
    inputs.layers_block_table = &model_input->layers_block_table;
    inputs.k_cache_blocks_base = &model_input->k_cache_blocks_base;
    inputs.v_cache_blocks_base = &model_input->v_cache_blocks_base;
    inputs.seq_len_host = &model_input->seq_len_host;
    inputs.forward_shape = &forward_context.forward_shape;
    inputs.atb_attention_attr = &model_input->atb_attention_attr;
    // inference on NPU with ATB
    STATUS_CHECK_RETURN(paged_attention_layer_->Forward(inputs, hidden_buffer_tensors_1));
  }
  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
#elif defined(ENABLE_KUNLUN)
  {
    // inference on Kunlun XPU — same tensor protocol as Ascend/ATB + optional QK-Norm weights
    KLLM_LOG_DEBUG << "PagedAttention::Forward (XPU) input_shape0=" << hidden_buffer_tensors_0[0].shape[0]
                   << " forward_shape0=" << forward_context.forward_shape.shape[0];
    PagedAttentionInputs inputs;
    inputs.qkv_tensor = &hidden_buffer_tensors_0[0];
    inputs.rotary_embedding_pos = &input_meta.rotary_embedding_pos;
    inputs.layers_slot_mapping = &model_input->layers_slot_mapping;
    inputs.layers_block_table = &model_input->layers_block_table;
    inputs.k_cache_blocks_base = &model_input->k_cache_blocks_base;
    inputs.v_cache_blocks_base = &model_input->v_cache_blocks_base;
    // PagedAttention is the single-token decode path; multi-token forwards go through FlashAttention.
    // seq_len is H2D'd into graph_seq_len_buf every single-token step in PrepareATBKVCache (both graph
    // and non-graph modes), so pass the device buffer here and avoid a per-layer compute-stream H2D.
    inputs.seq_len_host = &model_input->graph_seq_len_buf;
    inputs.forward_shape = &forward_context.forward_shape;
    inputs.atb_attention_attr = &model_input->atb_attention_attr;
    inputs.query_layernorm_weight = &query_layernorm_weight;
    inputs.key_layernorm_weight = &key_layernorm_weight;
    STATUS_CHECK_RETURN(paged_attention_layer_->Forward(inputs, hidden_buffer_tensors_1));
  }
  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
#endif
  return Status();
}

}  // namespace ksana_llm
