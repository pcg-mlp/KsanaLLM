/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v4/deepseek_v4_dspark.h"

#include <fmt/core.h>

#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta.h"
#include "models/deepseek_v4/deepseek_v4_model.h"
#include "profiler/profile_event.h"
#include "profiler/sched_event_tracer.h"
#include "utils/logger.h"

namespace ksana_llm {

DeepSeekV4DSparkLayer::DeepSeekV4DSparkLayer(const int layer_idx, LayerCreationContext& creation_context,
                                             ModelCreationConfig& model_creation_config,
                                             std::shared_ptr<DeepSeekV4DecoderLayer> decoder_layer, bool is_embed_layer)
    : layer_idx_(layer_idx),
      local_layer_idx_(creation_context.pipeline_config.ComputeLocalLayerIdx(
          layer_idx, static_cast<int>(model_creation_config.attn_config.model_config.num_layer))),
      hidden_dim_(model_creation_config.attn_config.model_config.hidden_units),
      layernorm_eps_(model_creation_config.layernorm_config.layernorm_eps),
      kv_cache_dtype_(model_creation_config.attn_config.kv_cache_dtype) {
  if (is_embed_layer) {
    num_target_layers_ = creation_context.model_config.dspark_config.dspark_target_layer_ids.size();
    main_norm_ =
        std::make_unique<Layernorm>(fmt::format("model.layers.{}.main_norm.weight", layer_idx),
                                    model_creation_config.attn_config.model_config.layernorm_eps, creation_context);
    main_proj_ = std::make_unique<Linear>(fmt::format("model.layers.{}.main_proj.weight", layer_idx), creation_context,
                                          model_creation_config.attn_config.model_config.quant_config.backend);
  }

  const std::string attn_prefix = fmt::format("model.layers.{}.self_attn", layer_idx);
  wkv_proj_ = std::make_unique<Linear>(attn_prefix + ".wkv.weight", creation_context,
                                       model_creation_config.attn_config.model_config.quant_config.backend);

  kv_norm_weight_ = creation_context.base_weight->GetModelWeights(attn_prefix + ".kv_norm.weight");

  // Inject-KV parameters setup.
  const size_t block_token_num = creation_context.runtime_config.attn_backend_config.block_token_num;
  KLLM_CHECK_WITH_INFO(block_token_num > 0 && (block_token_num & (block_token_num - 1)) == 0,
                       "DSparkInjectKV expects block_token_num power-of-two > 0");
  page_bits_ = static_cast<uint32_t>(__builtin_ctz(block_token_num));
  // block_token_num must in {2, 64, 256}
  KLLM_CHECK_WITH_INFO(page_bits_ == 1 || page_bits_ == 6 || page_bits_ == 8,
                       "DSparkInjectKV expects page_bits in {1, 6, 8}");
  cos_sin_cache_ptr_ = std::any_cast<void*>(model_creation_config.attn_config.cos_sin_cache_ptr);

  dspark_layer_ = std::make_unique<DSparkLayer>();
  dspark_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);

  decoder_layer_ = decoder_layer;
}

Status DeepSeekV4DSparkLayer::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  PROFILE_EVENT_SCOPE(DeepSeekV4DSparkLayer,
                      fmt::format("DeepSeekV4DSparkLayer_Forward_seq{}", residual_buffer[0].shape[0]),
                      forwarding_context.GetCurrentRank());
  RecordRequestSchedEvents(forwarding_context.GetBatchRequestSchedInfo(), forwarding_context.GetCurrentRank(),
                           forwarding_context.GetModelInput()->attn_dp_group_id_, "DSpark", RequestEventPhase::Begin);

  auto& dspark_hidden = forwarding_context.GetForwardingBuffers()->dspark_hidden_buffer_tensors[0];
  if (num_target_layers_ > 0) {
    // Convert concat'd hidden from selected main-decoder layers.
    KLLM_CHECK(dspark_hidden.shape[1] == num_target_layers_ * hidden_dim_);

    // main_proj into hidden_buffer_0: [T, num_target*hidden] -> [T, hidden].
    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
    STATUS_CHECK_RETURN(main_proj_->Forward(dspark_hidden, hidden_buffer_tensors_0[0]));

    // main_norm back into dspark_hidden: [T, hidden] -> [T, hidden].
    STATUS_CHECK_RETURN(main_norm_->Forward(hidden_buffer_tensors_0[0], dspark_hidden));
  }
  KLLM_CHECK(dspark_hidden.shape[1] == hidden_dim_);

  {
    // wkv_proj into hidden_buffer_tensors_0: [T, hidden] -> [T, head_dim]
    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
    STATUS_CHECK_RETURN(wkv_proj_->Forward(dspark_hidden, hidden_buffer_tensors_0));

    // Fused RMSNorm + RoPE + cache store.
    STATUS_CHECK_RETURN(InjectKV(hidden_buffer_tensors_0[0], forwarding_context));
  }

  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  STATUS_CHECK_RETURN(decoder_layer_->Forward(residual_buffer, is_multi_token_forward, forwarding_context));

  RecordRequestSchedEvents(forwarding_context.GetBatchRequestSchedInfo(), forwarding_context.GetCurrentRank(),
                           forwarding_context.GetModelInput()->attn_dp_group_id_, "DSpark", RequestEventPhase::End);
  return Status();
}

Status DeepSeekV4DSparkLayer::InjectKV(Tensor& kv_buffer, ForwardingContext& forwarding_context) {
  const size_t num_tokens = kv_buffer.shape[0];
  KLLM_CHECK_WITH_INFO(kv_buffer.dtype == TYPE_BF16, "DSparkInjectKV only supports bfloat16");
  KLLM_CHECK_WITH_INFO(num_tokens > 0, "DSparkInjectKV expects num_tokens > 0");

  const auto& model_input = forwarding_context.GetModelInput();
  auto* mqa_shared = model_input->shared_attention_meta->GetMqa();

  // Per-layer SWA cache base ptr lives at slots [2 + local_layer_idx_]
  void* kcache_ptr = mqa_shared->layer_swa_kv_cache_ptr_tensor.GetPtr<void*>()[2 + local_layer_idx_];
  const int64_t swa_block_size = mqa_shared->layer_swa_kv_cache_ptr_tensor.GetPtr<int64_t>()[1];
  // Use raw_out_loc/rotary_embedding_pos from Main forward
  const int32_t* last_raw_out_loc_ptr = mqa_shared->last_raw_out_loc_tensor.GetPtr<int32_t>();
  const int64_t* last_positions_ptr = mqa_shared->last_rotary_embedding_pos_tensor.GetPtr<int64_t>();

  STATUS_CHECK_RETURN(dspark_layer_->ForwardInjectKV(
      kv_buffer.GetPtr<void>(), kv_norm_weight_.GetPtr<void>(), kcache_ptr, last_positions_ptr, cos_sin_cache_ptr_,
      last_raw_out_loc_ptr, layernorm_eps_, num_tokens, swa_block_size, page_bits_, kv_cache_dtype_));
  return Status();
}

}  // namespace ksana_llm
