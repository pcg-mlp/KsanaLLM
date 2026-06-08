/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/basic/flash_mla_attention.h"

#include "layers/flash_mla_attention_layer.h"
#include "layers/flash_sparse_mla_attention_layer.h"
#include "models/base/attention_meta/attention_meta.h"
#include "models/base/attention_meta/attention_meta_builder.h"

namespace ksana_llm {

FlashMlaAttention::FlashMlaAttention(const size_t layer_idx, bool is_neox, const LayerCreationContext& creation_context,
                                     const AttentionCreationConfig& attn_config) {
  flash_mla_attention_layer_ = std::make_shared<FlashMlaAttentionLayer>();
  if (attn_config.model_config.use_dsa) {
    flash_sparse_mla_attention_layer_ = std::make_shared<FlashSparseMlaAttentionLayer>();
  }

  uint32_t qk_rope_head_dim = attn_config.model_config.mla_config.qk_rope_head_dim;
  uint32_t qk_nope_head_dim = attn_config.model_config.mla_config.qk_nope_head_dim;
  uint32_t q_lora_rank = attn_config.model_config.mla_config.q_lora_rank;
  uint32_t kv_lora_rank = attn_config.model_config.mla_config.kv_lora_rank;
  uint32_t v_head_dim = attn_config.model_config.mla_config.v_head_dim;

  // NOTE(karlluo): acsends's image g++ is 9.4.0, it do not support convert
  // from ‘<brace-enclosed initializer list>’ to ‘std::vector<std::any>’ so
  // we use push back to make it work.
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
  attention_param.push_back(attn_config.model_config.k_scales[layer_idx]);
  attention_param.push_back(attn_config.model_config.v_scales[layer_idx]);
  attention_param.push_back(attn_config.rotary_embedding);
  attention_param.push_back(attn_config.rope_theta);
  // new add for mla
  attention_param.push_back(qk_rope_head_dim);
  attention_param.push_back(qk_nope_head_dim);
  attention_param.push_back(q_lora_rank);
  attention_param.push_back(kv_lora_rank);
  attention_param.push_back(v_head_dim);
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
  // NOTE(karlluo): bool for
  // is_multi_token_forward
  flash_attention_param.push_back(true);
  flash_attention_param.push_back(attn_config.mrope_section_ptr);
  flash_attention_param.push_back(attn_config.model_config.enable_qk_pre_norm_before_rotary_pos);

  flash_mla_attention_layer_->Init(flash_attention_param, creation_context.runtime_config, creation_context.context,
                                   creation_context.rank);
  flash_mla_attention_layer_->SetWorkspaceBuffer(
      creation_context.workspace_mgr->GetWorkspace(flash_mla_attention_layer_->GetWorkspaceSize()));
  if (flash_sparse_mla_attention_layer_) {
    flash_sparse_mla_attention_layer_->Init(flash_attention_param, creation_context.runtime_config,
                                            creation_context.context, creation_context.rank);
    flash_sparse_mla_attention_layer_->SetWorkspaceBuffer(
        creation_context.workspace_mgr->GetWorkspace(flash_sparse_mla_attention_layer_->GetWorkspaceSize()));
  }

  // Initialize proj module
  kv_b_nope_proj_ = std::make_shared<Linear>(fmt::format("model.layers.{}.self_attn.kv_b_nope_proj.weight", layer_idx),
                                             creation_context, attn_config.model_config.quant_config.backend);
  // `kv_b_nope_proj_` and `v_head_proj_` have the same input and share the quantization results
  // We skip the quantization in `v_head_proj_`
  v_head_proj_ = std::make_shared<Linear>(fmt::format("model.layers.{}.self_attn.v_head_proj.weight", layer_idx),
                                          creation_context, attn_config.model_config.quant_config.backend,
                                          /*skip_quant*/ true);

  kv_b_nope_proj_weight_ = creation_context.base_weight->GetModelWeights(
      fmt::format("model.layers.{}.self_attn.kv_b_nope_proj.weight", layer_idx));
  v_head_proj_weight_ = creation_context.base_weight->GetModelWeights(
      fmt::format("model.layers.{}.self_attn.v_head_proj.weight", layer_idx));
}

Status FlashMlaAttention::Forward(const std::shared_ptr<ModelInput>& model_input,
                                  const AttentionForwardContext& attn_ctx, std::vector<Tensor>& k_buffer,
                                  std::vector<Tensor>& v_buffer, Tensor& context_q_nope_rope_tensor,
                                  Tensor& context_q_nope_tensor, Tensor& context_q_rope_tensor,
                                  Tensor& kv_buffer_tensor, Tensor& k_rope_buffer_tensor, Tensor& indices_tensor,
                                  std::vector<Tensor>& output_tensors) {
  auto& flash_mla = *model_input->flash_meta->GetMLA();
  auto& standard_shared = *model_input->shared_attention_meta->GetStandard();
  auto& mla_shared = *model_input->shared_attention_meta->GetMLA();
  if (indices_tensor.GetElementNumber() > 0) {
    FlashSparseMlaAttentionInputs sparse_inputs;
    sparse_inputs.rotary_embedding_pos = &model_input->flash_meta->rotary_embedding_pos;
    sparse_inputs.rotary_embedding_mask = &model_input->flash_meta->rotary_embedding_mask;
    sparse_inputs.q_pe = &context_q_rope_tensor;
    sparse_inputs.k_pe = &k_rope_buffer_tensor;
    sparse_inputs.q_nope = &context_q_nope_tensor;
    sparse_inputs.q = &k_buffer[0];
    sparse_inputs.compressed_kv = &kv_buffer_tensor;
    sparse_inputs.kv_list = &model_input->flash_meta->kv_list;
    sparse_inputs.prefix_offsets = &model_input->dp_input_prefix_uint64_tensor;
    sparse_inputs.seqlens_without_prefix = &model_input->dp_prefill_q_offset_uint64_tensor;
    sparse_inputs.cache_offset = &model_input->flash_meta->kv_cache_offset;
    sparse_inputs.layer_kv_cache = &standard_shared.layer_kv_cache_ptr;
    sparse_inputs.block_table = &model_input->flash_meta->block_table;
    sparse_inputs.tile_scheduler_metadata = &flash_mla.tile_scheduler_metadata;
    sparse_inputs.num_splits = &flash_mla.num_splits;
    sparse_inputs.indices = &indices_tensor;
    STATUS_CHECK_RETURN(flash_sparse_mla_attention_layer_->Forward(sparse_inputs, &output_tensors[0]));
    flash_mla.tile_scheduler_metadata.shape[2] = 1;
    return Status();
  } else {
    // When prefix cache hits, the following proj must be applied after fetching the prefix from kv cache
    if (/*dp_total_prefix_tokens*/ attn_ctx.forward_shape.shape[11] == 0) {
      // The quantized result of kv_buffer is stored in the workspace_buffer shared across layers,
      // and immediately required by v_head_proj, so no layer operations can be inserted between them
      STATUS_CHECK_RETURN(kv_b_nope_proj_->Forward(kv_buffer_tensor, output_tensors));
      STATUS_CHECK_RETURN(v_head_proj_->Forward(kv_buffer_tensor, v_buffer));
    }

    FlashMlaAttentionInputs mla_inputs{
        .dp_input_offset = &model_input->dp_input_offset_uint64_tensor,
        .dp_input_offset_host = &model_input->dp_input_offset_uint64_host,
        .kv_list = &model_input->flash_meta->kv_list,
        .dp_input_prefix = &model_input->dp_input_prefix_uint64_tensor,
        .dp_prefill_q_offset = &model_input->dp_prefill_q_offset_uint64_tensor,
        .dp_prefill_q_offset_host = &model_input->dp_prefill_q_offset_uint64_host,
        .chunk_kv_batch_starts_host = &mla_shared.chunk_kv_batch_starts_host,
        .chunk_kv_cu_extend_lens = &mla_shared.chunk_kv_cu_extend_lens_tensor,
        .chunk_kv_cu_seq_lens = &mla_shared.chunk_kv_cu_seq_lens_tensor,
        .kv_cache_offset = &model_input->flash_meta->kv_cache_offset,
        .rotary_embedding_pos = &model_input->flash_meta->rotary_embedding_pos,
        .rotary_embedding_mask = &model_input->flash_meta->rotary_embedding_mask,
        .dp_src_flexible_rotary_embedding_pos = &standard_shared.dp_src_flexible_rotary_embedding_pos,
        .dp_dst_flexible_rotary_embedding_pos = &standard_shared.dp_dst_flexible_rotary_embedding_pos,
        .dp_flexible_rotary_embedding_mask = &standard_shared.dp_flexible_rotary_embedding_mask,
        .dp_dst_flexible_kv_cache = &standard_shared.dp_dst_flexible_kv_cache_tensor,
        .dp_src_flexible_kv_cache = &standard_shared.dp_src_flexible_kv_cache_tensor,
        .dp_dst_flexible_token_idx = &standard_shared.dp_dst_flexible_token_idx_tensor,
        .dp_src_flexible_token_idx = &standard_shared.dp_src_flexible_token_idx_tensor,
        .dp_flexible_offset_uint64 = &standard_shared.dp_flexible_offset_uint64_tensor,
        .forward_shape = &attn_ctx.forward_shape,
        .q_nope_rope = &context_q_nope_rope_tensor,
        .kv_buffer = &kv_buffer_tensor,
        .k_rope_buffer = &k_rope_buffer_tensor,
        .kv_b_nope_proj_weight = &kv_b_nope_proj_weight_,
        .v_head_proj_weight = &v_head_proj_weight_,
        .k_buffer = &k_buffer[0],
        .v_buffer = &v_buffer[0],
    };
    return flash_mla_attention_layer_->Forward(mla_inputs, &output_tensors[0]);
  }
}

}  // namespace ksana_llm
