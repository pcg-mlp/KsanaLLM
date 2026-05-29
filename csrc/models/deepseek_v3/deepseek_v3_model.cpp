/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v3/deepseek_v3_model.h"

#include "device/device_utils.h"
#include "layers/assemble_tokens_hidden_layer.h"
#include "layers/concat_layer.h"
#include "layers/emb_lookup_layer.h"
#include "modules/basic/add.h"
#include "modules/basic/layernorm.h"
#include "profiler/profile_event.h"
#include "profiler/sched_event_tracer.h"

namespace ksana_llm {

// 用于MLA层张量并行处理的同步屏障
DeepSeekV3DecoderLayer::DeepSeekV3DecoderLayer(int layer_idx, bool is_moe, LayerCreationContext& creation_context,
                                               ModelCreationConfig& model_creation_config, MlaBuffers& mla_buffers,
                                               TensorBuffer* moe_buffer)
    : is_moe_(is_moe),
      enable_full_shared_expert_(creation_context.runtime_config.enable_full_shared_expert),
      layer_idx_(layer_idx),
      rank_(creation_context.rank),
      mla_buffers_(mla_buffers),
      moe_buffer_(moe_buffer) {
  MoeScaleNormMode moe_scale_norm_mode;
  if (model_creation_config.attn_config.model_config.mla_config.q_lora_rank != 0) {
    moe_scale_norm_mode = MoeScaleNormMode::RE_NORM;
  } else {
    moe_scale_norm_mode = MoeScaleNormMode::NO_NORM;
  }

  const std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  input_layernorm_ = std::make_shared<Layernorm>(
      layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);
  pre_attention_add_norm_ = std::make_shared<AddNorm>(
      layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);
  fused_all_reduce_norm_add_pre_attn_ = std::make_shared<FusedAllReduceNormAdd>(
      layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);

  add_ = std::make_shared<Add>(creation_context);

  post_attention_add_norm_ =
      std::make_shared<AddNorm>(layer_prefix + ".post_attention_layernorm.weight",
                                model_creation_config.layernorm_config.layernorm_eps, creation_context);
  fused_all_reduce_norm_add_post_attn_ =
      std::make_shared<FusedAllReduceNormAdd>(layer_prefix + ".post_attention_layernorm.weight",
                                              model_creation_config.layernorm_config.layernorm_eps, creation_context);

  if (is_moe) {
    shared_mlp_ = std::make_shared<TwoLayeredFFN>(layer_idx, creation_context, model_creation_config,
                                                  ".mlp.shared_expert.{}.weight");
    expert_gate_ = std::make_shared<Linear>(layer_prefix + ".mlp.gate.weight", creation_context,
                                            model_creation_config.attn_config.model_config.quant_config.backend);
    if (model_creation_config.attn_config.model_config.moe_config.use_e_score_correction_bias) {
      moe_ = std::make_shared<MoE>(
          layer_idx, layer_prefix + ".mlp.experts.up_gate_proj.weight", layer_prefix + ".mlp.experts.down_proj.weight",
          layer_prefix + ".mlp.gate.e_score_correction_bias", creation_context, moe_scale_norm_mode);
    } else {
      moe_ =
          std::make_shared<MoE>(layer_idx, layer_prefix + ".mlp.experts.up_gate_proj.weight",
                                layer_prefix + ".mlp.experts.down_proj.weight", creation_context, moe_scale_norm_mode);
    }
  } else {
    mlp_ = std::make_shared<TwoLayeredFFN>(layer_idx, creation_context, model_creation_config);
  }

  // mla should be init after linear, because mla will reuse workspace buffer which is created by linear layers
  mla_ = std::make_shared<MultiHeadLatentAttention>(layer_idx, /*is_neox*/ false, creation_context,
                                                    model_creation_config, mla_buffers_);

  tp_comm_ = std::make_shared<TpCommunicator>();
}

Status DeepSeekV3DecoderLayer::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                                       ForwardingContext& forwarding_context, bool need_add_residual_before_attn,
                                       bool need_add_residual_after_mlp) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);

  const size_t org_token_size = residual_buffer[0].shape[0];
  const size_t token_hidden_stat_bytes = residual_buffer[0].GetTotalBytes() / org_token_size;
  if (!need_add_residual_before_attn) {  // Adding the residual should have been done after mlp in the previous layer
                                         // for better performance.
    input_layernorm_->Forward(residual_buffer, hidden_buffer_tensors_0);
  } else if (forwarding_context.GetModelCommunicator() && enable_full_shared_expert_) {
    pre_attention_add_norm_->Forward({hidden_buffer_tensors_0[0], residual_buffer[0]}, hidden_buffer_tensors_0);
  } else {
    // Mlp/moe all reduce
    STATUS_CHECK_RETURN(fused_all_reduce_norm_add_pre_attn_->Forward(
        reduce_buffer_tensors, residual_buffer, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context,
        /*need_add_residual*/ true));
  }

  // Mla
  mla_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, tp_comm_, is_multi_token_forward, forwarding_context);
  if (forwarding_context.GetModelCommunicator() && enable_full_shared_expert_) {
    std::swap(reduce_buffer_tensors, hidden_buffer_tensors_0);
    post_attention_add_norm_->Forward({hidden_buffer_tensors_0[0], residual_buffer[0]}, hidden_buffer_tensors_0);
  } else {
    // Mla all reduce
    STATUS_CHECK_RETURN(fused_all_reduce_norm_add_post_attn_->Forward(
        reduce_buffer_tensors, residual_buffer, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context,
        /*need_add_residual*/ true));
  }

  const size_t dp_group_id = forwarding_context.GetModelInput()->attn_dp_group_id_;
  const int dp_token_offset = forwarding_context.GetModelInput()->attn_dp_group_offsets_[dp_group_id];
  const size_t dp_context_tokens = forwarding_context.GetModelInput()->dp_context_tokens;
  const size_t dp_decode_tokens = forwarding_context.GetModelInput()->dp_decode_tokens;
  const size_t dp_token_size = std::max(dp_context_tokens + dp_decode_tokens, 1ul);
  const size_t dp_bytes_offset = dp_token_offset * token_hidden_stat_bytes;
  const size_t dp_bytes = dp_token_size * token_hidden_stat_bytes;
  KLLM_LOG_DEBUG << fmt::format(
      "rank: {}, dp_group_id: {}, dp_token_offset: {}, dp_context_tokens: {}, dp_decode_tokens: {}, dp_token_size: {}, "
      "dp_bytes_offset: {}, dp_bytes: {}",
      forwarding_context.GetCurrentRank(), dp_group_id, dp_token_offset, dp_context_tokens, dp_decode_tokens,
      dp_token_size, dp_bytes_offset, dp_bytes);
  if (forwarding_context.GetModelCommunicator() && enable_full_shared_expert_) {
    std::swap(reduce_buffer_tensors, hidden_buffer_tensors_0);
    Stream& stream = forwarding_context.GetContext()->device->GetPrimaryComputeStream();
    MemcpyAsync(hidden_buffer_tensors_0[0].GetPtr<void>(), reduce_buffer_tensors[0].GetPtr<void>() + dp_bytes_offset,
                dp_bytes, MEMCPY_DEVICE_TO_DEVICE, stream);
    hidden_buffer_tensors_0[0].shape = {dp_token_size, reduce_buffer_tensors[0].shape[1]};
    reduce_buffer_tensors[0].shape = hidden_buffer_tensors_0[0].shape;
  }

  // Common mlp/moe
  AcquireMoeBuffers(forwarding_context);
  STATUS_CHECK_RETURN(
      CommonMlp(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));
  ReleaseMoeBuffers();

  // When using Expert-Parallel (EP) parallelism, Data-Parallel (DP) parallelism is enabled by default.
  // Since under EP, the MOE portion only computes the current DP data, so the data needs to be restored to the complete
  // length when MOE computation is finished.
  if (forwarding_context.GetModelCommunicator() && enable_full_shared_expert_) {
    std::swap(reduce_buffer_tensors, hidden_buffer_tensors_0);
    Stream& stream = forwarding_context.GetContext()->device->GetPrimaryComputeStream();
    MemcpyAsync(hidden_buffer_tensors_0[0].GetPtr<void>() + dp_bytes_offset, reduce_buffer_tensors[0].GetPtr<void>(),
                dp_bytes, MEMCPY_DEVICE_TO_DEVICE, stream);
    hidden_buffer_tensors_0[0].shape = {residual_buffer[0].shape[0], reduce_buffer_tensors[0].shape[1]};
    reduce_buffer_tensors[0].shape = hidden_buffer_tensors_0[0].shape;
    residual_buffer[0].shape = hidden_buffer_tensors_0[0].shape;
  }

  // Mlp/moe residual add
  // need_add_residual_after_mlp==false: residual is expected to be added before attn in the next layer for better
  // performance.
  if (need_add_residual_after_mlp) {
    if (forwarding_context.GetModelCommunicator() && enable_full_shared_expert_) {
      STATUS_CHECK_RETURN(add_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));
    } else {
      STATUS_CHECK_RETURN(fused_all_reduce_norm_add_pre_attn_->Forward(
          reduce_buffer_tensors, residual_buffer, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context,
          /*need_add_residual*/ true, /*need_apply_norm*/ false));
    }
  }

  return Status();
}

Status DeepSeekV3DecoderLayer::CommonMlp(std::vector<Tensor>& hidden_buffer_tensors_0,
                                         std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                         ForwardingContext& forwarding_context) {
  size_t seq_len = hidden_buffer_tensors_0[0].shape[0];
  size_t hidden_units = hidden_buffer_tensors_0[0].shape[1];

  PROFILE_EVENT_SCOPE(DS_CommonMlp_seq_len_,
                      fmt::format("DS_CommonMlp_seq_len_{}_hidden_units_{}", seq_len, hidden_units),
                      forwarding_context.GetCurrentRank());

  if (!is_moe_) {
    PROFILE_EVENT_SCOPE(CommonMlp, "CommonMlp", forwarding_context.GetCurrentRank());
    mlp_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context);
  } else {
    // Stage 1. Compute moe for mlp input from local nodes.
    CREATE_BUFFER_SCOPE(moe_buffer_tensors, moe_buffer_);
    auto& gated_buffer_ = moe_buffer_tensors;

    {
      PROFILE_EVENT_SCOPE(expert_gate, "expert_gate", forwarding_context.GetCurrentRank());
      STATUS_CHECK_RETURN(expert_gate_->Forward(hidden_buffer_tensors_0, gated_buffer_));
    }

    {
      PROFILE_EVENT_SCOPE(moe, "MOE", forwarding_context.GetCurrentRank());
      STATUS_CHECK_RETURN(
          moe_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0], reduce_buffer_tensors[0], moe_buffer_tensors));
    }

    {
      PROFILE_EVENT_SCOPE(CommonShareMlp, "CommonShareMlp", forwarding_context.GetCurrentRank());
      STATUS_CHECK_RETURN(shared_mlp_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward,
                                               forwarding_context));
    }

    if (forwarding_context.GetModelCommunicator() && !enable_full_shared_expert_) {
      PROFILE_EVENT_SCOPE(add_layer, "add_layer_with_comm", forwarding_context.GetCurrentRank());
      STATUS_CHECK_RETURN(add_->Forward(reduce_buffer_tensors[0], moe_buffer_tensors[0], reduce_buffer_tensors));
    } else {
      PROFILE_EVENT_SCOPE(add_layer, "add_layer", forwarding_context.GetCurrentRank());
      STATUS_CHECK_RETURN(add_->Forward(hidden_buffer_tensors_0[0], moe_buffer_tensors[0], hidden_buffer_tensors_0));
    }
  }
  return Status();
}

void DeepSeekV3DecoderLayer::AcquireMoeBuffers(ForwardingContext& forwarding_context) {
  // TODO(yancyliu): Get tensor from moe_buffer_
  // Reset its shape from batch_size and token_num, and then allocate tensor memory.
}

void DeepSeekV3DecoderLayer::ReleaseMoeBuffers() {
  // TODO(yancyliu): Get tensor from moe_buffer_, and then release its memory.
}

DeepSeekV3MtpLayer::DeepSeekV3MtpLayer(const int layer_idx, LayerCreationContext& creation_context,
                                       ModelCreationConfig& model_creation_config,
                                       std::shared_ptr<DeepSeekV3DecoderLayer> decoder_layer) {
  enorm_ = std::make_shared<Layernorm>(fmt::format("model.layers.{}.enorm.weight", layer_idx),
                                       model_creation_config.attn_config.model_config.layernorm_eps, creation_context);
  hnorm_ = std::make_shared<Layernorm>(fmt::format("model.layers.{}.hnorm.weight", layer_idx),
                                       model_creation_config.attn_config.model_config.layernorm_eps, creation_context);

  concat_layer_ = std::make_shared<ConcatLayer>();
  concat_layer_->Init({size_t{1}}, creation_context.runtime_config, creation_context.context, creation_context.rank);

  gather_layer_ = std::make_shared<AssembleTokensHiddenLayer>();
  gather_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);

  emb_lookup_layer_ = std::make_shared<EmbLookupLayer>();
  emb_lookup_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);

  eh_proj_ = std::make_shared<Linear>(fmt::format("model.layers.{}.eh_proj.weight", layer_idx), creation_context,
                                      model_creation_config.attn_config.model_config.quant_config.backend);

  decoder_layer_ = decoder_layer;

  tp_comm_ = std::make_shared<TpCommunicator>();
}

Status DeepSeekV3MtpLayer::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  RecordRequestSchedEvents(forwarding_context.GetBatchRequestSchedInfo(), forwarding_context.GetCurrentRank(),
                           forwarding_context.GetModelInput()->attn_dp_group_id_, "MTP", RequestEventPhase::Begin);
  auto& mtp_hidden_buffer = forwarding_context.GetForwardingBuffers()->mtp_hidden_buffer_tensors;
  const auto& model_input = forwarding_context.GetModelInput();
  {
    CREATE_BUFFER_SCOPE(shared_buffer, forwarding_context.GetForwardingBuffers()->shared_buffer);

    // Embedding Norm
    enorm_->Forward(residual_buffer, residual_buffer);

    // gather last token hidden
    STATUS_CHECK_RETURN(gather_layer_->Forward(
        {mtp_hidden_buffer[0], forwarding_context.GetModelInput()->nextn_hidden_idx_uint64_tensor}, shared_buffer));

    // last token hidden norm
    hnorm_->Forward(shared_buffer, mtp_hidden_buffer);

    // concat embedding_norm and hidden norm
    concat_layer_->Forward({residual_buffer[0], mtp_hidden_buffer[0]}, shared_buffer);

    // linear, no bias. hidden_units * 2 -> hidden_units
    STATUS_CHECK_RETURN(eh_proj_->Forward(shared_buffer[0], residual_buffer));
    tp_comm_->AllGather(residual_buffer[0], mtp_hidden_buffer[0], forwarding_context);
  }
  const bool is_multi_token_forward = model_input->multi_token_request_num > 0;
  STATUS_CHECK_RETURN(decoder_layer_->Forward(residual_buffer, is_multi_token_forward, forwarding_context,
                                              /* need_add_residual_before_attn */ false,
                                              /* need_add_residual_after_mlp */ true));

  RecordRequestSchedEvents(forwarding_context.GetBatchRequestSchedInfo(), forwarding_context.GetCurrentRank(),
                           forwarding_context.GetModelInput()->attn_dp_group_id_, "MTP", RequestEventPhase::End);
  return Status();
}

/**********************************************************
 * DeepSeekV3Model
 ***********************************************************/

DeepSeekV3Model::DeepSeekV3Model(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                                 std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
                                 std::shared_ptr<BaseModelCacheLayout> block_cache_layout)
    : CommonModel(model_config, runtime_config, rank, context),
      first_k_dense_replace_(model_config.moe_config.first_k_dense_replace) {
  ModelRunConfig model_run_config;
  model_run_config.position_encoding = PositionEncoding::ROPE;
  if (model_config.type == "glm_moe_dsa") {
    model_run_config.return_hidden_states_before_norm = false;
  }

  CommonModel::InitRunConfig(model_run_config, base_weight, block_cache_layout);
}

Status DeepSeekV3Model::CreateLayers(LayerCreationContext& creation_context,
                                     ModelCreationConfig& model_creation_config) {
  MultiHeadLatentAttention::CreateBuffers(CommonModel::GetBufferManager(), model_creation_config.attn_config,
                                          creation_context.runtime_config, mla_buffers_);
  const DataType weight_type = model_creation_config.attn_config.model_config.weight_data_type;
  const size_t max_token_num = creation_context.runtime_config.max_step_token_num;
  const size_t moe_buffer_size = max_token_num * model_creation_config.attn_config.model_config.hidden_units;
  moe_buffer_ = CommonModel::GetBufferManager()->CreateBufferTensor("moe_buffer_", {moe_buffer_size}, weight_type);

  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; ++layer_idx) {
    const bool is_moe = layer_idx >= first_k_dense_replace_;
    layers_[layer_idx] = std::make_shared<DeepSeekV3DecoderLayer>(layer_idx, is_moe, creation_context,
                                                                  model_creation_config, mla_buffers_, moe_buffer_);
  }

  if (pipeline_config_.lower_nextn_layer_idx >=
      static_cast<int>(model_creation_config.attn_config.model_config.num_layer)) {
    for (int layer_idx = pipeline_config_.lower_nextn_layer_idx; layer_idx <= pipeline_config_.upper_nextn_layer_idx;
         ++layer_idx) {
      const bool is_moe = layer_idx >= first_k_dense_replace_;
      layers_[layer_idx] = std::make_shared<DeepSeekV3DecoderLayer>(layer_idx, is_moe, creation_context,
                                                                    model_creation_config, mla_buffers_, moe_buffer_);
      // create nextn layer, give decoder layer
      nextn_layers_[layer_idx] =
          std::make_shared<DeepSeekV3MtpLayer>(layer_idx, creation_context, model_creation_config, layers_[layer_idx]);
    }
    nextn_layer_idx_ = pipeline_config_.lower_nextn_layer_idx;

    // Create MTP-specific shared_head.norm (shared across all MTP steps)
    if (context_->global->IsSamplingNode()) {
      nextn_shared_head_norm_ = std::make_unique<Layernorm>(
          fmt::format("model.layers.{}.shared_head.norm.weight", pipeline_config_.lower_nextn_layer_idx),
          model_creation_config.attn_config.model_config.layernorm_eps, creation_context);
    }
  }
  return Status();
}

Status DeepSeekV3Model::LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode) {
  PROFILE_EVENT_SCOPE(DS_LayerForward_, "DS_LayerForward", forwarding_context.GetCurrentRank());
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;

  if (run_mode == RunMode::kMain) {
    std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);
    for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
         layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
      STATUS_CHECK_RETURN(layers_[layer_idx]->Forward(
          residual_buffer, is_multi_token_forward, forwarding_context,
          /* need_add_residual_before_attn */ layer_idx != forwarding_context.GetPipelineConfig().lower_layer_idx,
          /* need_add_residual_after_mlp */ layer_idx == forwarding_context.GetPipelineConfig().upper_layer_idx));
    }
  } else if (run_mode == RunMode::kNextN && !nextn_layers_.empty()) {
    forwarding_context.SetIsForwardingLayers(false);  // Don't record ForwardingLayers event
    std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);
    for (int layer_idx = nextn_layer_idx_; layer_idx < nextn_layer_idx_ + kNextnDecoderLayerNum; ++layer_idx) {
      STATUS_CHECK_RETURN(nextn_layers_[nextn_layer_idx_]->Forward(residual_buffer, forwarding_context));
    }

    nextn_layer_idx_ += kNextnDecoderLayerNum;
    if (nextn_layer_idx_ >= pipeline_config_.lower_nextn_layer_idx + model_config_.num_nextn_predict_layers) {
      nextn_layer_idx_ = pipeline_config_.lower_nextn_layer_idx;
    }
  }

  if (forwarding_context.GetForwardingBuffers()->runtime_config.enable_full_shared_expert) {
    // EP 开启时，需要清除其他 DP 的脏数据
    std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);
    const size_t org_token_size = residual_buffer[0].shape[0];
    const size_t token_hidden_stat_bytes = residual_buffer[0].GetTotalBytes() / org_token_size;
    const size_t dp_group_id = forwarding_context.GetModelInput()->attn_dp_group_id_;
    const int dp_token_offset = forwarding_context.GetModelInput()->attn_dp_group_offsets_[dp_group_id];
    const size_t dp_token_size =
        forwarding_context.GetModelInput()->dp_context_tokens + forwarding_context.GetModelInput()->dp_decode_tokens;

    if (dp_token_offset > 0) {
      MemsetAsync(residual_buffer[0].GetPtr<void>(), 0, dp_token_offset * token_hidden_stat_bytes,
                  forwarding_context.GetContext()->device->GetPrimaryComputeStream());
    }

    if (dp_token_offset + dp_token_size < org_token_size) {
      MemsetAsync(residual_buffer[0].GetPtr<void>() + (dp_token_offset + dp_token_size) * token_hidden_stat_bytes, 0,
                  (org_token_size - dp_token_offset - dp_token_size) * token_hidden_stat_bytes,
                  forwarding_context.GetContext()->device->GetPrimaryComputeStream());
    }
  }
  return Status();
}

}  // namespace ksana_llm
