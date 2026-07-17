/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/attention/c4_indexer.h"

#include <fmt/core.h>

#include "layers/flash_sparse_mla_indexer_layer.h"
#include "layers/paged_sparse_mla_indexer_layer.h"
#include "layers/topk_layer.h"
#include "models/base/attention_meta/attention_meta.h"
#include "models/base/attention_meta/attention_meta_builder.h"
#include "models/base/attention_meta/c4_indexer_attention_meta.h"
#include "models/base/attention_meta/c4_indexer_shared_attention_meta.h"
#include "profiler/profile_event.h"
#include "utils/logger.h"

namespace ksana_llm {

C4Indexer::C4Indexer(int layer_idx, const LayerCreationContext& creation_context,
                     const ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx) {
  const auto& attn_config = model_creation_config.attn_config;
  const auto& model_config = attn_config.model_config;
  const auto& runtime_config = creation_context.runtime_config;

  KLLM_CHECK_WITH_INFO(
      layer_idx >= 0 && layer_idx < static_cast<int>(model_config.dsv4_attn_config.compress_ratios.size()),
      "layer_idx out of bounds for compress_ratios array");
  const uint32_t compress_ratio = model_config.dsv4_attn_config.compress_ratios[layer_idx];
  KLLM_CHECK_WITH_INFO(compress_ratio == 4, "C4Indexer requires compress_ratio == 4");

  // Initialize dimensions from model config
  KLLM_CHECK_WITH_INFO(model_config.use_dsa, "C4Indexer requires use_dsa == true");
  n_heads_ = model_config.dsa_config.index_n_heads;
  head_dim_ = model_config.dsa_config.index_head_dim;
  index_topk_ = model_config.dsa_config.index_topk;

  // block_size refers to the number of tokens in one block
  block_size_ = runtime_config.attn_backend_config.block_token_num;

  const std::string layer_prefix = fmt::format("model.layers.{}.self_attn.indexer", layer_idx);
  const auto& linear_compute_backend = model_config.quant_config.backend;

  // Initialize linear layers
  wq_b_ = std::make_unique<Linear>(layer_prefix + ".wq_b.weight", creation_context, linear_compute_backend);

  weights_proj_ =
      std::make_unique<Linear>(layer_prefix + ".weights_proj.weight", creation_context, linear_compute_backend);

  // Internal compressor with is_in_indexer=true, compress_ratio=4
  compressor_ =
      std::make_unique<Compressor>(layer_idx_, /*is_in_indexer=*/true, creation_context, model_creation_config);

  // Initialize flash and paged indexer layers
  flash_sparse_mla_indexer_layer_ = std::make_unique<FlashSparseMlaIndexerLayer>();
  paged_sparse_mla_indexer_layer_ = std::make_unique<PagedSparseMlaIndexerLayer>();

  std::vector<std::any> indexer_param;
  indexer_param.push_back(static_cast<int>(n_heads_));   // n_heads
  indexer_param.push_back(static_cast<int>(head_dim_));  // head_dim
  indexer_param.push_back(
      static_cast<int>(attn_config.model_config.dsv4_attn_config.qk_rope_head_dim));  // rope_head_dim
  indexer_param.push_back(static_cast<int>(index_topk_));                             // index_topk
  indexer_param.push_back(static_cast<int>(block_size_));                             // block_size (kv block size)
  indexer_param.push_back(128);                                     // quant_block_size (quantization block size)
  indexer_param.push_back(attn_config.position_encoding);           // rotary_embedding type
  indexer_param.push_back(attn_config.compress_rope_theta);         // rope_theta
  indexer_param.push_back(attn_config.compress_cos_sin_cache_ptr);  // cos/sin cache
  indexer_param.push_back(attn_config.model_config.rope_scaling_factor_config);  // rope scaling
  indexer_param.push_back(static_cast<int>(attn_config.max_batch_size));         // max_batch_size
  indexer_param.push_back(attn_config.idx);                                      // local_layer_idx
  indexer_param.push_back(static_cast<int>(attn_config.layer_num_on_node));      // layer_num_on_node
  indexer_param.push_back(model_config.dsa_config.indexer_rope_interleave);      // rope_interleave
  indexer_param.push_back(model_config.quant_config.scale_fmt);                  // scale_fmt
  indexer_param.push_back(compress_ratio);                                       // compress_ratio
  flash_sparse_mla_indexer_layer_->Init(indexer_param, creation_context.runtime_config, creation_context.context,
                                        creation_context.rank);
  flash_sparse_mla_indexer_layer_->SetWorkspaceBuffer(
      creation_context.workspace_mgr->GetWorkspace(flash_sparse_mla_indexer_layer_->GetWorkspaceSize()));

  indexer_param.push_back(static_cast<int>(attn_config.max_decode_tokens_per_req));  // max_decode_tokens_per_req
  paged_sparse_mla_indexer_layer_->Init(indexer_param, creation_context.runtime_config, creation_context.context,
                                        creation_context.rank);
  paged_sparse_mla_indexer_layer_->SetWorkspaceBuffer(
      creation_context.workspace_mgr->GetWorkspace(paged_sparse_mla_indexer_layer_->GetWorkspaceSize()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward
// ─────────────────────────────────────────────────────────────────────────────

Status C4Indexer::Forward(const Tensor& x, const Tensor& qr, const Tensor& workspace,
                          std::vector<Tensor>& q_indexer_tensors, Tensor& topk_indices,
                          ForwardingContext& forwarding_context) {
  const int rank = forwarding_context.GetCurrentRank();
  PROFILE_EVENT_SCOPE(c4_indexer_forward, "c4_indexer_forward", rank);

  // x.shape = [total_tokens, hidden_dim]
  const size_t total_tokens = x.shape[0];
  // Used to locate unused workspace buffers
  // Assume the given workspace is sufficient since n_heads and head_dim in indexer is small
  size_t workspace_offset = 0;

  // Step 1: Key compression via compressor (wkv_gate projection + compress)
  Tensor kv_compressed_output = workspace.GetView({total_tokens, head_dim_}, workspace_offset);
  workspace_offset += kv_compressed_output.GetElementNumber();
  STATUS_CHECK_RETURN(compressor_->Forward(x, workspace, kv_compressed_output, forwarding_context));

  // Step 2: Compute weights = weights_proj(x)
  // Input: x [total_tokens, hidden_dim], Output: weight [total_tokens, n_heads]
  std::vector<Tensor> weights_tensors{workspace.GetView({total_tokens, n_heads_}, workspace_offset)};
  workspace_offset += weights_tensors[0].GetElementNumber();
  STATUS_CHECK_RETURN(weights_proj_->Forward(x, weights_tensors));

  // Step 3: Query projection q = wq_b(qr)
  // Input: qr [total_tokens, q_lora_rank], Output: q [total_tokens, n_heads * head_dim]
  STATUS_CHECK_RETURN(wq_b_->Forward(qr, q_indexer_tensors));

  // Step 4: Process flash and paged attention separately for top-k token selection
  // Use large enough workspace for quantization and TopK V2 scratch.
  const size_t topkv2_workspace_bytes = kWrokspaceIntsPerBatch * sizeof(int32_t);
  const size_t topkv2_workspace_elements =
      (topkv2_workspace_bytes + workspace.GetDTypeSize() - 1) / workspace.GetDTypeSize();
  size_t quant_workspace_size = std::max(n_heads_ * head_dim_, topkv2_workspace_elements);
  Tensor quant_workspace = workspace.GetView({total_tokens * quant_workspace_size}, workspace_offset);
  auto& attn_ctx = forwarding_context.GetAttentionForwardContext();
  const auto& model_input = forwarding_context.GetModelInput();

  auto& c4_shared_indexer =
      static_cast<C4IndexerSharedAttentionMeta&>(*model_input->shared_attention_meta->GetIndexer());
  auto& flash_c4_indexer = static_cast<C4IndexerAttentionMeta&>(*model_input->flash_meta->GetIndexer());

  const size_t dp_context_tokens = c4_shared_indexer.dp_context_tokens;
  const size_t dp_context_kv_tokens = c4_shared_indexer.dp_context_kv_tokens;
  const size_t dp_decode_tokens = c4_shared_indexer.dp_decode_tokens;
  const size_t q_stride = n_heads_ * head_dim_;
  const size_t k_stride = head_dim_;

  if (dp_context_tokens > 0) {
    // Flash attention for prefill tokens (first dp_context_tokens)
    Tensor context_q_indexer = q_indexer_tensors[0].GetView({dp_context_tokens, q_stride});
    Tensor context_k_indexer = kv_compressed_output.GetView({dp_context_kv_tokens, k_stride});
    Tensor context_weights = weights_tensors[0].GetView({dp_context_tokens, n_heads_});

    FlashSparseMlaIndexerLayer::FlashSparseMlaIndexerParams flash_params{
        .q = &context_q_indexer,
        .k = &context_k_indexer,
        .weights = &context_weights,
        .quant_workspace = &quant_workspace,
        .rotary_embedding_pos = &model_input->flash_meta->rotary_embedding_pos,
        .store_out_loc = &c4_shared_indexer.gathered_c4_out_loc_tensor,
        .seq_len_offset = &model_input->dp_input_offset_uint64_tensor,
        .block_table = &model_input->flash_meta->block_table,
        .cu_seq_len_start = &flash_c4_indexer.cu_seq_len_start_tensor,
        .cu_seq_len_end = &flash_c4_indexer.cu_seq_len_end_tensor,
        .batch_indices = &flash_c4_indexer.batch_indices_tensor,
        .layer_kv_cache_ptr = &model_input->shared_attention_meta->GetDSA()->layer_indexer_kv_cache_ptr,
        .chunk_token_starts_host = &c4_shared_indexer.chunk_token_starts_host,
        .chunk_kv_token_ranges_host = &c4_shared_indexer.chunk_kv_token_ranges_host,
        .chunk_max_compressed_kv_lens_host = &c4_shared_indexer.chunk_max_compressed_kv_lens_host,
        .batch_size = static_cast<int>(attn_ctx.forward_shape.shape[7]),
        .total_prefix_tokens = static_cast<int>(attn_ctx.forward_shape.shape[11]),
    };
    Tensor context_output = topk_indices.GetView({dp_context_tokens, index_topk_});
    auto* flash_typed = static_cast<FlashSparseMlaIndexerLayer*>(flash_sparse_mla_indexer_layer_.get());
    STATUS_CHECK_RETURN(flash_typed->Forward(flash_params, &context_output));
  }

  if (dp_decode_tokens > 0) {
    // Paged attention for decode tokens (after dp_context_tokens)
    // Process page_dual and page_single sequentially, since requests are sorted by token_num in descending order
    size_t skip_tokens = dp_context_tokens;
    size_t skip_kv_tokens = dp_context_kv_tokens;
    for (size_t i = 0; i < model_input->page_metas.size(); ++i) {
      auto& page_c4_indexer = static_cast<C4IndexerAttentionMeta&>(*model_input->page_metas[i]->GetIndexer());
      const size_t current_tokens = page_c4_indexer.total_dp_input_ids_len;
      const size_t current_kv_tokens = page_c4_indexer.total_dp_compressed_input_ids_len;
      KLLM_LOG_DEBUG << "page_metas[" << i << "]: max_compressed_seq_len=" << page_c4_indexer.max_compressed_seq_len;

      Tensor current_q_indexer = q_indexer_tensors[0].GetView({current_tokens, q_stride}, skip_tokens * q_stride);
      Tensor current_k_indexer = kv_compressed_output.GetView({current_kv_tokens, k_stride}, skip_kv_tokens * k_stride);
      Tensor current_weights = weights_tensors[0].GetView({current_tokens, n_heads_}, skip_tokens * n_heads_);

      auto& page_dsa = *model_input->page_metas[i]->GetDSA();
      PagedSparseMlaIndexerLayer::PagedSparseMlaIndexerParams paged_params{
          .q = &current_q_indexer,
          .k = &current_k_indexer,
          .weights = &current_weights,
          .quant_workspace = &quant_workspace,
          .rotary_embedding_pos = &model_input->page_metas[i]->rotary_embedding_pos,
          .store_out_loc = &c4_shared_indexer.gathered_c4_out_loc_tensor,
          .input_length = &page_c4_indexer.c4_seq_lens_tensor,
          .expanded_input_length = &page_c4_indexer.expanded_c4_seq_lens_tensor,
          .block_table = &model_input->page_metas[i]->block_table,
          .expanded_block_table = &page_dsa.expanded_block_table,
          .cu_seq_len_start = &page_c4_indexer.cu_seq_len_start_tensor,
          .cu_seq_len_end = &page_c4_indexer.cu_seq_len_end_tensor,
          .layer_kv_cache_ptr = &model_input->shared_attention_meta->GetDSA()->layer_indexer_kv_cache_ptr,
          .schedule_meta = &page_dsa.paged_schedule_meta,
          .max_compressed_seq_len = static_cast<int64_t>(page_c4_indexer.max_compressed_seq_len),
          .sched_meta_initialized = (page_dsa.paged_schedule_meta.shape[2] != 0),
      };
      Tensor current_output = topk_indices.GetView({current_tokens, index_topk_}, skip_tokens * index_topk_);
      auto* paged_typed = static_cast<PagedSparseMlaIndexerLayer*>(paged_sparse_mla_indexer_layer_.get());
      STATUS_CHECK_RETURN(paged_typed->Forward(paged_params, &current_output));
      page_dsa.paged_schedule_meta.shape[2] = 1;

      skip_tokens += current_tokens;
      skip_kv_tokens += current_kv_tokens;
    }
  }

  // Correctly set the output shape
  topk_indices.shape = {total_tokens, index_topk_};

  KLLM_LOG_DEBUG << fmt::format("C4Indexer Forward completed: layer={}", layer_idx_);

  return Status();
}

}  // namespace ksana_llm
