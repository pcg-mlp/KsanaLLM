/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/attention/sparse_mla_indexer.h"

#include "layers/flash_sparse_mla_indexer_layer.h"
#include "layers/paged_sparse_mla_indexer_layer.h"
#include "profiler/profile_event.h"
#include "utils/logger.h"
#include "utils/tensor.h"

namespace ksana_llm {

SparseMlaIndexer::SparseMlaIndexer(int layer_idx, LayerCreationContext& creation_context,
                                   ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx) {
  auto& attn_config = model_creation_config.attn_config;
  auto& model_config = attn_config.model_config;
  auto& runtime_config = creation_context.runtime_config;

  // Initialize dimensions from model config
  n_heads_ = model_config.dsa_config.index_n_heads;
  head_dim_ = model_config.dsa_config.index_head_dim;
  index_topk_ = model_config.dsa_config.index_topk;

  // block_size指的是一个block 里面有多少个token，区别cache block size：表示一个block的字节个数
  block_size_ = runtime_config.attn_backend_config.block_token_num;

  const std::string layer_prefix = fmt::format("model.layers.{}.self_attn", layer_idx);
  const auto& linear_compute_backend = model_config.quant_config.backend;

  // Initialize linear layers
  wq_b_ = std::make_shared<Linear>(layer_prefix + ".indexer.wq_b.weight", creation_context, linear_compute_backend);

  wk_ = std::make_shared<Linear>(layer_prefix + ".indexer.wk.weight", creation_context, linear_compute_backend);

  weights_proj_ =
      std::make_shared<Linear>(layer_prefix + ".indexer.weights_proj.weight", creation_context, linear_compute_backend);

  // Initialize layer norm
  k_norm_ = std::make_shared<Layernorm>(layer_prefix + ".indexer.k_norm.weight",
                                        model_creation_config.layernorm_config.layernorm_eps, creation_context,
                                        layer_prefix + ".indexer.k_norm.bias");

  // Initialize flash and paged indexer layers with local layer index
  flash_sparse_mla_indexer_layer_ = std::make_unique<FlashSparseMlaIndexerLayer>();
  paged_sparse_mla_indexer_layer_ = std::make_unique<PagedSparseMlaIndexerLayer>();

  // acsends's image g++ is 9.4.0, it do not support convert
  // from '<brace-enclosed initializer list>' to 'std::vector<std::any>' so
  // we use push back to make it work.
  std::vector<std::any> indexer_param;
  indexer_param.push_back(static_cast<int>(attn_config.model_config.hidden_units));                 // 0: dim
  indexer_param.push_back(static_cast<int>(n_heads_));                                              // 1: n_heads
  indexer_param.push_back(static_cast<int>(head_dim_));                                             // 2: head_dim
  indexer_param.push_back(static_cast<int>(attn_config.model_config.mla_config.qk_rope_head_dim));  // 3: rope_head_dim
  indexer_param.push_back(static_cast<int>(index_topk_));                                           // 4: index_topk
  indexer_param.push_back(static_cast<int>(block_size_));  // 5: block_size (kv block size)
  indexer_param.push_back(128);                            // 6: quant_block_size (quantization block size)
  indexer_param.push_back(attn_config.position_encoding);  // 7: rotary_embedding type
  indexer_param.push_back(attn_config.rope_theta);         // 8: rope_theta
  indexer_param.push_back(attn_config.cos_sin_cache_ptr);  // 9: cos/sin cache
  indexer_param.push_back(attn_config.model_config.rope_scaling_factor_config);  // 10: rope scaling
  indexer_param.push_back(static_cast<int>(attn_config.max_batch_size));         // 11: max_batch_size
  indexer_param.push_back(attn_config.max_position_embeddings);                  // 12: max_seq_len
  indexer_param.push_back(attn_config.idx);                                      // 13: local_layer_idx
  indexer_param.push_back(model_config.dsa_config.indexer_rope_interleave);      // 14: rope_interleave
  indexer_param.push_back(model_config.quant_config.scale_fmt);                  // 15: scale_fmt
  flash_sparse_mla_indexer_layer_->Init(indexer_param, creation_context.runtime_config, creation_context.context,
                                        creation_context.rank);
  flash_sparse_mla_indexer_layer_->SetWorkspaceBuffer(
      creation_context.workspace_mgr->GetWorkspace(flash_sparse_mla_indexer_layer_->GetWorkspaceSize()));

  indexer_param.push_back(static_cast<int>(attn_config.layer_num_on_node));          // 15: layer_num_on_node
  indexer_param.push_back(static_cast<int>(attn_config.max_decode_tokens_per_req));  // 16: max_decode_tokens_per_req
  paged_sparse_mla_indexer_layer_->Init(indexer_param, creation_context.runtime_config, creation_context.context,
                                        creation_context.rank);
  paged_sparse_mla_indexer_layer_->SetWorkspaceBuffer(
      creation_context.workspace_mgr->GetWorkspace(paged_sparse_mla_indexer_layer_->GetWorkspaceSize()));
}

Status SparseMlaIndexer::Forward(const Tensor& x, const Tensor& qr, const Tensor& workspace, const bool context_use_dsa,
                                 Tensor& topk_indices, ForwardingContext& forwarding_context) {
  const int rank = forwarding_context.GetCurrentRank();

  PROFILE_EVENT_SCOPE(mla_forward, "sparse_mla_index_forward", rank);
  // x.shape = [total_tokens, hidden_dim]
  const size_t total_tokens = x.shape[0];
  const size_t hidden_dim = x.shape[1];

  KLLM_LOG_DEBUG << fmt::format("SparseMlaIndexer Forward: layer={}, total_tokens={}, hidden_dim={}", layer_idx_,
                                total_tokens, hidden_dim);

  // Step 1: Key projection k = wk(x)
  // Input: x [total_tokens, dim]
  // Output: k [total_tokens, head_dim]
  std::vector<Tensor> k_indexer_tensors{workspace.GetView({total_tokens, head_dim_})};
  {
    PROFILE_EVENT_SCOPE(wk_forward, "indexer_wk", rank);
    STATUS_CHECK_RETURN(wk_->Forward(x, k_indexer_tensors));
  }
  // Step 2: Key normalization
  {
    PROFILE_EVENT_SCOPE(k_norm_forward, "indexer_k_norm", rank);
    STATUS_CHECK_RETURN(k_norm_->Forward(k_indexer_tensors, k_indexer_tensors));
  }

  // Calculate token counts
  const auto model_input = forwarding_context.GetModelInput();
  const size_t dp_context_tokens = model_input->dp_context_tokens;
  const size_t dp_decode_tokens = model_input->dp_decode_tokens;

  // Step 3: Compute weights = weights_proj(x)
  // Input: x [total_tokens, dim]
  // Output: weight [total_tokens, n_local_heads]
  std::vector<Tensor> weights_tensors{workspace.GetView({total_tokens, n_heads_}, total_tokens * head_dim_)};
  {
    PROFILE_EVENT_SCOPE(weights_proj_forward, "indexer_weights_proj", rank);
    if (context_use_dsa || dp_context_tokens == 0) {
      STATUS_CHECK_RETURN(weights_proj_->Forward(x, weights_tensors));
    } else if (dp_decode_tokens > 0) {
      Tensor decode_x = x.GetView({dp_decode_tokens, x.shape[1]}, dp_context_tokens * x.shape[1]);
      std::vector<Tensor> decode_weights_tensors = {
          weights_tensors[0].GetView({dp_decode_tokens, n_heads_}, dp_context_tokens * n_heads_)};
      STATUS_CHECK_RETURN(weights_proj_->Forward(decode_x, decode_weights_tensors));
      weights_tensors[0].shape = decode_weights_tensors[0].shape;
    }
  }

  // Step 4: Query projection q = wq_b(qr)
  // Input: qr [total_tokens, q_lora_rank]
  // Output: q [total_tokens, n_local_heads * head_dim]
  std::vector<Tensor> q_indexer_tensors{x.GetView({total_tokens, n_heads_, head_dim_})};
  {
    PROFILE_EVENT_SCOPE(wq_b_forward, "indexer_wq_b", rank);
    if (context_use_dsa || dp_context_tokens == 0) {
      STATUS_CHECK_RETURN(wq_b_->Forward(qr, q_indexer_tensors));
    } else if (dp_decode_tokens > 0) {
      Tensor decode_qr = qr.GetView({dp_decode_tokens, qr.shape[1]}, dp_context_tokens * qr.shape[1]);
      std::vector<Tensor> decode_q_indexer_tensors = {q_indexer_tensors[0].GetView(
          {dp_decode_tokens, n_heads_, head_dim_}, dp_context_tokens * n_heads_ * head_dim_)};
      STATUS_CHECK_RETURN(wq_b_->Forward(decode_qr, decode_q_indexer_tensors));
      q_indexer_tensors[0].shape = decode_q_indexer_tensors[0].shape;
    }
  }

  // Step 5: Process flash and paged attention separately
  // Use large enough workspace
  Tensor quant_workspace =
      workspace.GetView({total_tokens * n_heads_ * head_dim_}, total_tokens * (n_heads_ + head_dim_));
  auto& attn_ctx = forwarding_context.GetAttentionForwardContext();

  if (dp_context_tokens > 0) {
    // Flash attention for prefill tokens (first dp_context_tokens)
    Tensor context_q_indexer = q_indexer_tensors[0].GetView({dp_context_tokens, n_heads_, head_dim_});
    Tensor context_k_indexer = k_indexer_tensors[0].GetView({dp_context_tokens, head_dim_});
    Tensor context_weights = (context_use_dsa ? weights_tensors[0].GetView({dp_context_tokens, n_heads_}) : Tensor{});

    std::vector<Tensor> context_output_tensors = {topk_indices.GetView({dp_context_tokens, index_topk_})};
    STATUS_CHECK_RETURN(flash_sparse_mla_indexer_layer_->Forward(
        {context_q_indexer, context_k_indexer, context_weights, quant_workspace,
         model_input->flash_input.rotary_embedding_pos, model_input->flash_input.rotary_embedding_mask,
         model_input->flash_input.indexer_kv_list, model_input->flash_input.kv_cache_offset,
         model_input->dp_prefill_q_offset_uint64_tensor, model_input->dp_input_prefix_uint64_tensor,
         model_input->dp_input_offset_uint64_tensor, model_input->flash_input.block_table,
         model_input->flash_input.cur_seq_len_start, model_input->flash_input.cur_seq_len_end,
         model_input->layer_indexer_kv_cache_ptr, attn_ctx.forward_shape},
        context_output_tensors));
  }

  if (dp_decode_tokens > 0) {
    // Paged attention for decode tokens (after dp_context_tokens)
    // Process page_dual and page_single sequentially
    // Page_dual is placed before page_single, since requests are sorted by token_num in descending order
    size_t skip_tokens = 0;
    for (const auto& page_input : model_input->page_inputs) {
      const size_t current_tokens = page_input.total_dp_input_ids_len;
      // Offset tensors by `dp_context_tokens + skip_tokens`
      const size_t total_offset = dp_context_tokens + skip_tokens;

      Tensor current_q_indexer =
          q_indexer_tensors[0].GetView({current_tokens, n_heads_, head_dim_}, total_offset * n_heads_ * head_dim_);
      Tensor current_k_indexer = k_indexer_tensors[0].GetView({current_tokens, head_dim_}, total_offset * head_dim_);
      Tensor current_weights = weights_tensors[0].GetView({current_tokens, n_heads_}, total_offset * n_heads_);

      // Offset output tensor for current page
      std::vector<Tensor> current_output_tensors = {
          topk_indices.GetView({current_tokens, index_topk_}, total_offset * index_topk_)};

      STATUS_CHECK_RETURN(paged_sparse_mla_indexer_layer_->Forward(
          {current_q_indexer, current_k_indexer, current_weights, quant_workspace, page_input.rotary_embedding_pos,
           page_input.rotary_embedding_mask, page_input.input_length, page_input.expanded_input_length,
           page_input.indexer_kv_list, page_input.kv_cache_offset, page_input.block_table,
           page_input.expanded_block_table, page_input.cur_seq_len_start, page_input.cur_seq_len_end,
           model_input->layer_indexer_kv_cache_ptr, page_input.paged_schedule_meta},
          current_output_tensors));
      skip_tokens += current_tokens;
    }
    for (auto& page_input : model_input->page_inputs) {
      page_input.paged_schedule_meta.shape[2] = 1;  // mark meta as initialized
    }
  }

  // Correctly set the output shape
  topk_indices.shape = {total_tokens, index_topk_};

  KLLM_LOG_DEBUG << fmt::format("SparseMlaIndexer Forward completed: layer={}", layer_idx_);

  return Status();
}

}  // namespace ksana_llm
