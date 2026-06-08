/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/attention/dsv4_mqa.h"

#include <fmt/core.h>
#include <string>

#include "layers/flash_dsv4_attention_layer.h"
#include "layers/paged_dsv4_attention_layer.h"
#include "models/base/attention_meta/attention_meta.h"
#include "models/base/attention_meta/attention_meta_builder.h"
#include "profiler/profile_event.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
Dsv4Mqa::Dsv4Mqa(int layer_idx, bool is_neox, bool add_qkv_bias, bool use_qk_norm,
                 LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config,
                 Dsv4AttnBuffer& dsv4_attn_buffer)
    : dsv4_attn_buffer_(dsv4_attn_buffer) {
  layer_idx_ = layer_idx;
  const std::string attn_prefix = fmt::format("model.layers.{}.self_attn", layer_idx);

  const auto& model_config = model_creation_config.attn_config.model_config;
  const auto& dsv4_attn_config = model_config.dsv4_attn_config;
  const auto& linear_backend = model_config.quant_config.backend;
  const float layernorm_eps = model_creation_config.layernorm_config.layernorm_eps;
  const size_t atp_size = creation_context.runtime_config.parallel_basic_config.attn_tensor_parallel_size;

  auto& attn_config = model_creation_config.attn_config;
  // attn_config.idx must be set first
  attn_config.idx = layer_idx - creation_context.pipeline_config.lower_layer_idx;
  if (layer_idx >= static_cast<int>(model_creation_config.attn_config.model_config.num_layer)) {
    attn_config.idx = creation_context.pipeline_config.upper_layer_idx -
                      creation_context.pipeline_config.lower_layer_idx + layer_idx -
                      model_creation_config.attn_config.model_config.num_layer + 1;
  }

  // ── Dimension fields ────────────────────────────────────────────────────────
  // head_dim = qk_nope_head_dim + qk_rope_head_dim (== v_head_dim for MQA)
  head_dim_ = model_config.size_per_head;
  q_lora_rank_ = dsv4_attn_config.q_lora_rank;
  o_lora_rank_ = dsv4_attn_config.o_lora_rank;
  KLLM_CHECK_WITH_INFO(model_config.head_num % atp_size == 0, "head_num must be divisible by atp_size");
  n_local_heads_ = model_config.head_num / atp_size;
  KLLM_CHECK_WITH_INFO(dsv4_attn_config.o_groups % atp_size == 0, "o_groups must be divisible by atp_size");
  o_local_groups_ = dsv4_attn_config.o_groups / atp_size;

  // ── compress_ratio ───────────────────────────────────────────────────────────
  KLLM_CHECK_WITH_INFO(
      layer_idx >= 0 && layer_idx < static_cast<int>(model_config.dsv4_attn_config.compress_ratios.size()),
      "layer_idx out of bounds for compress_ratios array");
  compress_ratio_ = dsv4_attn_config.compress_ratios[layer_idx];
  KLLM_CHECK_WITH_INFO(compress_ratio_ == 0 || compress_ratio_ == 4 || compress_ratio_ == 128,
                       "Dsv4Attn requires compress_ratio in {0, 4, 128}");

  // ── Q path weights ───────────────────────────────────────────────────────────
  // wqkv_a: [hidden_size → q_lora_rank + head_dim]
  wqkv_a_proj_ = std::make_unique<Linear>(attn_prefix + ".wqkv_a.weight", creation_context, linear_backend);

  // Parallel q_norm + kv_norm: strided RMSNorm
  qkv_a_norm_ =
      std::make_unique<Layernorm>(attn_prefix + ".q_norm.weight", attn_prefix + ".kv_norm.weight", layernorm_eps,
                                  creation_context, static_cast<int32_t>(q_lora_rank_ + head_dim_));

  // wq_b: [q_lora_rank → n_heads * head_dim], TP column-parallel
  wq_b_proj_ = std::make_unique<Linear>(attn_prefix + ".wq_b.weight", creation_context, linear_backend);

  // ── DSV4 Attention layers ─────────────────────────────────────────────────────
  // Initialize flash (prefill) and paged (decode) DSV4 attention layers
  // following the same pattern as FlashMlaAttention/PagedMlaAttention.
  std::vector<std::any> attention_param;
  attention_param.push_back(attn_config.model_config.quant_config.method);  // quant method
  attention_param.push_back(attn_config.model_config.layernorm_eps);        // layernorm eps
  attention_param.push_back(attn_config.model_config.use_qk_norm);          // use_qk_norm
  attention_param.push_back(attn_config.idx);                               // local layer idx
  attention_param.push_back(attn_config.layer_num_on_node);                 // layer_num_on_node
  attention_param.push_back(attn_config.max_position_embeddings);           // max_position_embeddings
  attention_param.push_back(attn_config.head_num_per_tp);                   // head_num_per_tp
  attention_param.push_back(attn_config.num_kv_heads_per_tp);               // num_kv_heads_per_tp
  attention_param.push_back(attn_config.size_per_head);                     // size_per_head
  attention_param.push_back(attn_config.stride_size);       // stride_size (head_num + num_kv_heads * 2) * size_per_head
  attention_param.push_back(attn_config.tensor_para_size);  // tensor_para_size
  attention_param.push_back(attn_config.kv_cache_dtype);    // kv_cache_dtype
  attention_param.push_back(attn_config.model_config.k_scales[layer_idx]);  // k_scale
  attention_param.push_back(attn_config.model_config.v_scales[layer_idx]);  // v_scale
  attention_param.push_back(attn_config.rotary_embedding);                  // rotary_embedding
  attention_param.push_back(compress_ratio_ > 0 ? attn_config.compress_rope_theta
                                                : attn_config.rope_theta);  // rope_theta
  // MLA params
  attention_param.push_back(static_cast<uint32_t>(dsv4_attn_config.qk_rope_head_dim));  // qk_rope_head_dim
  attention_param.push_back(
      static_cast<uint32_t>(model_config.size_per_head - dsv4_attn_config.qk_rope_head_dim));  // qk_nope_head_dim
  attention_param.push_back(static_cast<uint32_t>(dsv4_attn_config.q_lora_rank));              // q_lora_rank
  attention_param.push_back(static_cast<uint32_t>(0));                           // kv_lora_rank (N/A for MQA)
  attention_param.push_back(static_cast<uint32_t>(model_config.size_per_head));  // v_head_dim
  // End MLA params
  attention_param.push_back(is_neox);                        // is_neox
  attention_param.push_back(attn_config.position_encoding);  // position_encoding
  attention_param.push_back(compress_ratio_ > 0 ? attn_config.compress_cos_sin_cache_ptr
                                                : attn_config.cos_sin_cache_ptr);  // cos_sin_cache
  if (compress_ratio_ > 0) {
    attention_param.push_back(attn_config.model_config.rope_scaling_factor_config);  // rope_scaling (yarn)
  } else {
    attention_param.push_back(
        RoPEScalingFactor{.use_deepseek_rope = true});  // rope_scaling (default, with deepseek rope kernel)
  }
  attention_param.push_back(attn_config.max_batch_size);  // max_batch_size
  // Temperature tuning
  attention_param.push_back(attn_config.model_config.attn_temperature_tuning);  // attn_temperature_tuning
  attention_param.push_back(attn_config.model_config.attn_scale);               // attn_scale
  attention_param.push_back(attn_config.model_config.floor_scale);              // floor_scale

  // Flash attention (prefill)
  std::vector<std::any> flash_attention_param = attention_param;
  flash_attention_param.push_back(true);  // is_multi_token
  flash_attention_param.push_back(attn_config.mrope_section_ptr);
  flash_attention_param.push_back(attn_config.model_config.enable_qk_pre_norm_before_rotary_pos);
  flash_attention_param.push_back(compress_ratio_);  // dsv4 specific (append last)

  flash_dsv4_attention_layer_ = std::make_unique<FlashDsv4AttentionLayer>();
  flash_dsv4_attention_layer_->Init(flash_attention_param, creation_context.runtime_config, creation_context.context,
                                    creation_context.rank);
  flash_dsv4_attention_layer_->SetWorkspaceBuffer(
      creation_context.workspace_mgr->GetWorkspace(flash_dsv4_attention_layer_->GetWorkspaceSize()));

  // Paged attention (decode)
  std::vector<std::any> paged_attention_param = attention_param;
  paged_attention_param.push_back(false);    // is_multi_token
  paged_attention_param.push_back(nullptr);  // mrope_section_ptr
  paged_attention_param.push_back(attn_config.model_config.enable_qk_pre_norm_before_rotary_pos);
  paged_attention_param.push_back(compress_ratio_);  // dsv4 specific (append last)

  paged_dsv4_attention_layer_ = std::make_unique<PagedDsv4AttentionLayer>();
  paged_dsv4_attention_layer_->Init(paged_attention_param, creation_context.runtime_config, creation_context.context,
                                    creation_context.rank);
  paged_dsv4_attention_layer_->SetWorkspaceBuffer(
      creation_context.workspace_mgr->GetWorkspace(paged_dsv4_attention_layer_->GetWorkspaceSize()));

  // ── Attention sink weight ─────────────────────────────────────────────────────
  attn_sink_weight_ = creation_context.base_weight->GetModelWeights(attn_prefix + ".attn_sink");

  // ── Output path weights ──────────────────────────────────────────────────────
  // wo_a: group-wise Bmm, weight shape [n_local_groups, o_lora_rank, head_dim * n_local_heads / n_local_groups]
  wo_a_bmm_ = std::make_unique<Bmm>(attn_prefix + ".wo_a.weight", creation_context, linear_backend);

  // wo_b: [n_groups * o_lora_rank → hidden_size], TP row-parallel
  wo_b_proj_ = std::make_unique<Linear>(attn_prefix + ".wo_b.weight", creation_context, linear_backend);

  // ── Compressor / C4Indexer (production modules from modules/attention/) ──────
  if (compress_ratio_ > 0) {
    compressor_ =
        std::make_unique<Compressor>(layer_idx_, /*is_in_indexer=*/false, creation_context, model_creation_config);
  }
  if (compress_ratio_ == 4) {
    c4_indexer_ = std::make_unique<C4Indexer>(layer_idx_, creation_context, model_creation_config);
  }

  // Save the finalised local layer index (attn_config.idx is fully resolved by now).
  local_layer_idx_ = attn_config.idx;
}

Status Dsv4Mqa::CreateBuffers(BufferManager* buffer_mgr, const AttentionCreationConfig& attn_config,
                              const RuntimeConfig& runtime_config, Dsv4AttnBuffer& dsv4_attn_buffer) {
  const DataType weight_type = attn_config.model_config.weight_data_type;
  const size_t max_token_num = runtime_config.max_step_token_num;
  const size_t head_dim = attn_config.model_config.size_per_head;
  const size_t index_n_heads = attn_config.model_config.dsa_config.index_n_heads;
  const size_t index_head_dim = attn_config.model_config.dsa_config.index_head_dim;

  const size_t kv_compressed_output_buffer_size = max_token_num * head_dim;
  dsv4_attn_buffer.kv_compressed_output_buffer = buffer_mgr->CreateBufferTensor(
      "dsv4_attn_buffer.kv_compressed_output_buffer", {kv_compressed_output_buffer_size}, weight_type);

  const size_t q_indexer_buffer_size = max_token_num * index_n_heads * index_head_dim;
  dsv4_attn_buffer.q_indexer_buffer =
      buffer_mgr->CreateBufferTensor("dsv4_attn_buffer.q_indexer_buffer", {q_indexer_buffer_size}, weight_type);

  return Status();
}

Status Dsv4Mqa::Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                        ForwardingContext& forwarding_context) {
  const int rank = forwarding_context.GetCurrentRank();
  const auto& x_tensors = hidden_buffer_tensors_0;  // [T, hidden_size]
  const size_t total_tokens = x_tensors[0].shape[0];

  PROFILE_EVENT_SCOPE(dsv4_mqa_forward, fmt::format("dsv4_mqa_layer_seq_{}_hidden_units_{}", total_tokens, head_dim_),
                      rank);

  // ── Step 1: Q path + KV path ──────────────────────────────────────────
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);

  // ── Fused wqkv_a path ───────────────────────────────────────────────────
  // wqkv_a: [T, hidden_size] → [T, q_lora_rank + head_dim] (in hidden_buffer_tensors_1)
  std::vector<Tensor>& qkv_a_tensors = hidden_buffer_tensors_1;
  STATUS_CHECK_RETURN(wqkv_a_proj_->Forward(x_tensors[0], qkv_a_tensors[0]));

  // Parallel RMSNorm: reads strided q_a and kv_a from wqkv_a output, writes contiguous
  //   q_a_input:  ptr = base,               shape = {T, q_lora_rank}, stride = q_lora_rank + head_dim
  //   kv_a_input: ptr = base + q_lora_rank, shape = {T, head_dim},    stride = q_lora_rank + head_dim
  // Place normed q_a/kv_a in reduce_buffer_tensors
  std::vector<Tensor> q_a_tensors{reduce_buffer_tensors[0].GetView({total_tokens, static_cast<size_t>(q_lora_rank_)})};
  Tensor kv_a_tensor{reduce_buffer_tensors[0].GetView({total_tokens, static_cast<size_t>(head_dim_)},
                                                      q_a_tensors[0].GetElementNumber())};
  STATUS_CHECK_RETURN(qkv_a_norm_->ForwardParallel(
      qkv_a_tensors[0].GetView({total_tokens, static_cast<size_t>(q_lora_rank_)}),
      qkv_a_tensors[0].GetView({total_tokens, static_cast<size_t>(head_dim_)}, q_lora_rank_), q_a_tensors[0],
      kv_a_tensor));

  // Now hidden_buffer_tensors_1 is free

  const auto& model_input = forwarding_context.GetModelInput();
  auto& mqa_shared = *model_input->shared_attention_meta->GetMqa();

  // ── C4 Indexer (compress_ratio == 4) ─────────────────────────────────────
  auto& topk_indices_tensor = mqa_shared.c4_block_indices_tensor;
  if (c4_indexer_) {
    // Pass x [T, hidden_size] and q_a [T, q_lora_rank] (after normed) to c4_indexer.
    // Use hidden_buffer_tensors_1[0] as workspace for c4_indexer intermediates.
    // Output kv_compressed_output_tensor [T, head_dim] and topk_indices [T, index_topk] stored in pre-allocated buffer.
    CREATE_BUFFER_SCOPE(q_indexer_tensors, dsv4_attn_buffer_.q_indexer_buffer);
    STATUS_CHECK_RETURN(c4_indexer_->Forward(x_tensors[0], q_a_tensors[0],
                                             /*workspace*/ hidden_buffer_tensors_1[0], q_indexer_tensors,
                                             topk_indices_tensor, forwarding_context));
  }

  // ── Compressor (compress_ratio != 0) ─────────────────────────────────────
  CREATE_BUFFER_SCOPE(kv_compressed_output_tensors, dsv4_attn_buffer_.kv_compressed_output_buffer);
  Tensor& kv_compressed_output_tensor = kv_compressed_output_tensors[0];
  if (compressor_) {
    // Pass x [T, hidden_size] to compressor.
    // Use hidden_buffer_tensors_1[0] as workspace for compressor intermediates.
    // Output kv_compressed_output_tensor [T, head_dim] stored in pre-allocated buffer.
    STATUS_CHECK_RETURN(compressor_->Forward(x_tensors[0], /*workspace*/ hidden_buffer_tensors_1[0],
                                             kv_compressed_output_tensor, forwarding_context));
  }

  // Now hidden_buffer_tensors_0 and hidden_buffer_tensors_1 are free

  // ── wq_b: q_lora → Q ─────────────────────────────────────────────────────
  // wq_b: [T, q_lora_rank] → [T, n_local_heads * head_dim] (in hidden_buffer_tensors_1)
  std::vector<Tensor>& q_tensors = hidden_buffer_tensors_1;
  STATUS_CHECK_RETURN(wq_b_proj_->Forward(q_a_tensors[0], q_tensors[0]));

  // Now hidden_buffer_tensors_0 is free

  // ── Step 2: DSV4 MQA attention ─────────────────────────────────────────────
  // DSV4 attention layers handle RoPE, KV cache store, and sparse MQA internally.
  Tensor& attn_output = hidden_buffer_tensors_0[0];
  const size_t dp_context_tokens = model_input->dp_context_tokens;
  const size_t dp_decode_tokens = model_input->dp_decode_tokens;
  const size_t q_stride = n_local_heads_ * head_dim_;
  const size_t kv_stride = head_dim_;

  if (dp_context_tokens > 0) {
    auto& flash_mqa = static_cast<DeepSeekV4FlashAttentionMeta&>(*model_input->flash_meta).mqa;

    if (flash_mqa.has_non_contiguous_swa) {
      // Compute the byte offset of this layer's KV data within each pool block.
      // After InitSwaBasePointersIfNeeded: real_layer_ptrs[li] = pool_base + layer_offset.
      const void* swa_pool_base = model_input->flash_meta->dp_reqs[0]->swa_block_base_addr;
      void** real_layer_ptrs = mqa_shared.layer_swa_kv_cache_ptr_tensor.GetPtr<void*>() + 2;
      const size_t layer_offset =
          reinterpret_cast<size_t>(real_layer_ptrs[local_layer_idx_]) - reinterpret_cast<size_t>(swa_pool_base);
      const size_t per_layer = static_cast<size_t>(mqa_shared.per_layer_swa_block_size);

      auto& compute_stream = forwarding_context.GetContext()->device->GetPrimaryComputeStream();
      // Pre-copy: pool (layer portion) → buffer  (src_offset = layer's byte offset in pool block)
      if (!flash_mqa.swa_pre_copy_src_host.empty()) {
        BatchMemoryD2D(flash_mqa.swa_pre_copy_src_tensor, flash_mqa.swa_pre_copy_dst_tensor, layer_offset,
                       /*dst_offset*/ 0, per_layer, compute_stream);
      }

      STATUS_CHECK_RETURN(RunDsv4Attention<FlashDsv4AttentionLayer>(
          flash_dsv4_attention_layer_.get(), *model_input->flash_meta, flash_mqa, q_tensors[0], kv_a_tensor,
          attn_output, topk_indices_tensor, model_input->shared_attention_meta->GetStandard()->layer_kv_cache_ptr,
          mqa_shared.layer_swa_cache_buffer_ptr_tensor, dp_context_tokens,
          /*token_offset*/ 0, q_stride, kv_stride));

      // Post-copy: buffer → pool (layer portion)  (dst_offset = layer's byte offset in pool block)
      if (!flash_mqa.swa_post_copy_src_host.empty()) {
        BatchMemoryD2D(flash_mqa.swa_post_copy_src_tensor, flash_mqa.swa_post_copy_dst_tensor,
                       /*src_offset*/ 0, layer_offset, per_layer, compute_stream);
      }
    } else {
      STATUS_CHECK_RETURN(RunDsv4Attention<FlashDsv4AttentionLayer>(
          flash_dsv4_attention_layer_.get(), *model_input->flash_meta, flash_mqa, q_tensors[0], kv_a_tensor,
          attn_output, topk_indices_tensor, model_input->shared_attention_meta->GetStandard()->layer_kv_cache_ptr,
          mqa_shared.layer_swa_kv_cache_ptr_tensor, dp_context_tokens,
          /*token_offset*/ 0, q_stride, kv_stride));
    }
  }

  if (dp_decode_tokens > 0) {
    size_t skip_tokens = dp_context_tokens;

    for (size_t i = 0; i < model_input->page_metas.size(); i++) {
      auto& paged_mqa = static_cast<DeepSeekV4PagedAttentionMeta&>(*model_input->page_metas[i]).mqa;
      const size_t current_tokens = model_input->page_metas[i]->total_dp_input_ids_len;

      STATUS_CHECK_RETURN(RunDsv4Attention<PagedDsv4AttentionLayer>(
          paged_dsv4_attention_layer_.get(), *model_input->page_metas[i], paged_mqa, q_tensors[0], kv_a_tensor,
          attn_output, topk_indices_tensor, model_input->shared_attention_meta->GetStandard()->layer_kv_cache_ptr,
          mqa_shared.layer_swa_kv_cache_ptr_tensor, current_tokens, skip_tokens, q_stride, kv_stride));

      skip_tokens += current_tokens;
    }
  }

  // Restore output shape after attention (in hidden_buffer_tensors_0)
  // Reshape for group-wise Bmm: [T, n_local_heads * head_dim] → [T, n_local_groups, d_per_group]
  const size_t d_per_group = n_local_heads_ * head_dim_ / o_local_groups_;
  attn_output.shape = {total_tokens, o_local_groups_, d_per_group};

  // Now hidden_buffer_tensors_1 and reduce_buffer_tensors are free

  // ── Step 3: Output path ─────────────────────────────────────────────────────
  // wo_a Bmm: [T, n_local_groups, d_per_group] × [n_local_groups, d_per_group, o_lora_rank]
  //         → [T, n_local_groups, o_lora_rank]
  STATUS_CHECK_RETURN(wo_a_bmm_->Forward(/*attn_output*/ hidden_buffer_tensors_0[0], hidden_buffer_tensors_1[0]));

  // Flatten for wo_b: [T, n_local_groups * o_lora_rank]
  hidden_buffer_tensors_1[0].shape = {total_tokens, o_local_groups_ * o_lora_rank_};

  // wo_b: [T, n_local_groups * o_lora_rank] → [T, hidden_size]
  // (in reduce_buffer_tensors if TP mode, otherwise in hidden_buffer_tensors_0)
  if (forwarding_context.GetModelCommunicator()) {
    STATUS_CHECK_RETURN(wo_b_proj_->Forward(hidden_buffer_tensors_1[0], reduce_buffer_tensors[0]));
  } else {
    STATUS_CHECK_RETURN(wo_b_proj_->Forward(hidden_buffer_tensors_1[0], hidden_buffer_tensors_0[0]));
  }

  return Status();
}

template <typename LayerT>
Status Dsv4Mqa::RunDsv4Attention(LayerT* attention_layer, StandardAttentionMeta& model_input_info,
                                 MqaAttentionMeta& mqa_input_info, Tensor& q_tensor, Tensor& kv_tensor,
                                 Tensor& attn_output, Tensor& topk_indices_tensor, Tensor& layer_kv_cache_ptr,
                                 Tensor& layer_swa_kv_cache_ptr_tensor, size_t token_count, size_t token_offset,
                                 size_t q_stride, size_t kv_stride) {
  Tensor current_q = q_tensor.GetView({token_count, q_stride}, token_offset * q_stride);
  Tensor current_kv = kv_tensor.GetView({token_count, kv_stride}, token_offset * kv_stride);
  Tensor current_output = attn_output.GetView({token_count, q_stride}, token_offset * q_stride);

  auto& model_mla = *model_input_info.GetMLA();
  Tensor& tile_sched_meta = compress_ratio_ == 0   ? model_mla.tile_scheduler_metadata
                            : compress_ratio_ == 4 ? mqa_input_info.c4_tile_scheduler_metadata
                                                   : mqa_input_info.c128_tile_scheduler_metadata;
  Tensor& num_splits = compress_ratio_ == 0   ? model_mla.num_splits
                       : compress_ratio_ == 4 ? mqa_input_info.c4_num_splits
                                              : mqa_input_info.c128_num_splits;

  // Declare here so lifetime spans past the if-blocks below
  Tensor c4_extra_indices_view;

  Dsv4AttentionInputs inputs;
  inputs.rotary_embedding_pos = &model_input_info.rotary_embedding_pos;
  inputs.q = &current_q;
  inputs.kv = &current_kv;
  inputs.layer_swa_kv_cache = &layer_swa_kv_cache_ptr_tensor;
  inputs.tile_scheduler_metadata = &tile_sched_meta;
  inputs.num_splits = &num_splits;
  inputs.attn_sink = &attn_sink_weight_;
  inputs.raw_out_loc = &mqa_input_info.raw_out_loc_tensor;
  inputs.swa_indices = &mqa_input_info.swa_block_indices_tensor;
  if (compress_ratio_ > 0) {
    inputs.layer_extra_kv_cache = &layer_kv_cache_ptr;
    if (compress_ratio_ == 4) {
      c4_extra_indices_view = topk_indices_tensor.GetView({token_count, topk_indices_tensor.shape[1]},
                                                          token_offset * topk_indices_tensor.shape[1]);
      inputs.extra_indices = &c4_extra_indices_view;
    } else {  // compress_ratio_ == 128
      inputs.extra_indices = &mqa_input_info.c128_block_indices_tensor;
      inputs.extra_topk_length = &mqa_input_info.c128_topk_lengths_tensor;
    }
  }
  STATUS_CHECK_RETURN(attention_layer->Forward(inputs, &current_output));
  tile_sched_meta.shape[2] = 1;  // Mark tile_scheduler_metadata as initialized

  return Status();
}

// Explicit template instantiations
template Status Dsv4Mqa::RunDsv4Attention<FlashDsv4AttentionLayer>(FlashDsv4AttentionLayer*, StandardAttentionMeta&,
                                                                   MqaAttentionMeta&, Tensor&, Tensor&, Tensor&,
                                                                   Tensor&, Tensor&, Tensor&, size_t, size_t, size_t,
                                                                   size_t);
template Status Dsv4Mqa::RunDsv4Attention<PagedDsv4AttentionLayer>(PagedDsv4AttentionLayer*, StandardAttentionMeta&,
                                                                   MqaAttentionMeta&, Tensor&, Tensor&, Tensor&,
                                                                   Tensor&, Tensor&, Tensor&, size_t, size_t, size_t,
                                                                   size_t);

}  // namespace ksana_llm
