/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/hy_v3/hy_v3_model.h"

#include "layers/assemble_tokens_hidden_layer.h"
#include "layers/concat_layer.h"
#include "models/hy_v3/hy_v3_mtp_draft_model.h"
#include "profiler/profile_event.h"
#include "profiler/sched_event_tracer.h"

namespace ksana_llm {

HyV3DecoderLayer::HyV3DecoderLayer(int layer_idx, bool is_moe, TensorBuffer* moe_buffer,
                                   LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx), is_moe_(is_moe), moe_buffer_(moe_buffer) {
  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  add_ = std::make_shared<Add>(creation_context);
  tp_comm_ = std::make_shared<TpCommunicator>();

  input_layernorm_ = std::make_shared<Layernorm>(
      layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);
  post_attention_layernorm_ =
      std::make_shared<Layernorm>(layer_prefix + ".post_attention_layernorm.weight",
                                  model_creation_config.layernorm_config.layernorm_eps, creation_context);

  // GQA + pre-RoPE QK-Norm, no QKV bias. Pre-RoPE norm is driven by
  // model_config.enable_qk_pre_norm_before_rotary_pos (set in model_instance.cpp);
  // use_qk_norm here loads the query/key layernorm weights.
  const bool is_neox = true;
  const bool add_qkv_bias = false;
  const bool use_qk_norm = true;
  mha_ = std::make_shared<MultiHeadAttention>(layer_idx, is_neox, add_qkv_bias, use_qk_norm, creation_context,
                                              model_creation_config);

  if (is_moe) {
    expert_gate_ = std::make_shared<Linear>(layer_prefix + ".mlp.gate.weight", creation_context,
                                            model_creation_config.attn_config.model_config.quant_config.backend);
    moe_ = std::make_shared<MoE>(
        layer_idx, layer_prefix + ".mlp.experts.up_gate_proj.weight", layer_prefix + ".mlp.experts.down_proj.weight",
        layer_prefix + ".mlp.gate.e_score_correction_bias", creation_context, MoeScaleNormMode::NO_NORM);
    shared_mlp_ = std::make_shared<TwoLayeredFFN>(layer_idx, creation_context, model_creation_config,
                                                  ".mlp.shared_expert.{}.weight");
  } else {
    mlp_ = std::make_shared<TwoLayeredFFN>(layer_idx, creation_context, model_creation_config);
  }
}

Status HyV3DecoderLayer::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                                 ForwardingContext& forwarding_context) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);

  // Pre-Norm before attention.
  input_layernorm_->Forward(residual_buffer[0], hidden_buffer_tensors_0[0]);

  STATUS_CHECK_RETURN(
      mha_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Attn residual add.
  STATUS_CHECK_RETURN(add_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));

  // Pre-Norm before MLP/MoE.
  post_attention_layernorm_->Forward(residual_buffer[0], hidden_buffer_tensors_0[0]);

  STATUS_CHECK_RETURN(
      ForwardMlp(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // MLP/MoE residual add.
  STATUS_CHECK_RETURN(add_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));
  return Status();
}

Status HyV3DecoderLayer::ForwardMlp(std::vector<Tensor>& hidden_buffer_tensors_0,
                                    std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                    ForwardingContext& forwarding_context) {
  if (!is_moe_) {
    mlp_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context);
    return Status();
  }

  CREATE_BUFFER_SCOPE(moe_buffer_tensors, moe_buffer_);
  auto& gated_buffer_ = reduce_buffer_tensors;

  // Router logits + routed experts.
  STATUS_CHECK_RETURN(expert_gate_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0]));
  STATUS_CHECK_RETURN(
      moe_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0], reduce_buffer_tensors[0], moe_buffer_tensors));

  // Preserve routed output before the shared expert reuses moe_buffer_tensors.
  MemcpyAsync(reduce_buffer_tensors[0].GetPtr<void>(), moe_buffer_tensors[0].GetPtr<void>(),
              moe_buffer_tensors[0].GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
              forwarding_context.GetContext()->device->GetPrimaryComputeStream());
  reduce_buffer_tensors[0].shape = moe_buffer_tensors[0].shape;
  reduce_buffer_tensors[0].dtype = moe_buffer_tensors[0].dtype;

  // Shared expert over the SAME (post-attention-norm) input.
  STATUS_CHECK_RETURN(
      shared_mlp_->Forward(hidden_buffer_tensors_0, moe_buffer_tensors, is_multi_token_forward, forwarding_context));

  // TwoLayeredFFN writes to reduce_buffer_tensors (== moe_buffer_tensors here) when a communicator
  // exists (TP>1), else into hidden_buffer_tensors_0 via its internal swap.
  const bool has_model_communicator = (forwarding_context.GetModelCommunicator() != nullptr);
  auto& shared_expert_output = has_model_communicator ? moe_buffer_tensors[0] : hidden_buffer_tensors_0[0];
  if (has_model_communicator) {
    STATUS_CHECK_RETURN(add_->Forward(reduce_buffer_tensors[0], shared_expert_output, reduce_buffer_tensors));
  } else {
    STATUS_CHECK_RETURN(add_->Forward(reduce_buffer_tensors[0], shared_expert_output, hidden_buffer_tensors_0));
  }
  return Status();
}

/**********************************************************
 * HyV3MtpLayer
 ***********************************************************/

HyV3MtpLayer::HyV3MtpLayer(int layer_idx, LayerCreationContext& creation_context,
                           ModelCreationConfig& model_creation_config,
                           std::shared_ptr<HyV3DecoderLayer> decoder_layer) {
  enorm_ = std::make_shared<Layernorm>(fmt::format("model.layers.{}.enorm.weight", layer_idx),
                                       model_creation_config.attn_config.model_config.layernorm_eps, creation_context);
  hnorm_ = std::make_shared<Layernorm>(fmt::format("model.layers.{}.hnorm.weight", layer_idx),
                                       model_creation_config.attn_config.model_config.layernorm_eps, creation_context);

  concat_layer_ = std::make_shared<ConcatLayer>();
  concat_layer_->Init({size_t{1}}, creation_context.runtime_config, creation_context.context, creation_context.rank);

  gather_layer_ = std::make_shared<AssembleTokensHiddenLayer>();
  gather_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);

  eh_proj_ = std::make_shared<Linear>(fmt::format("model.layers.{}.eh_proj.weight", layer_idx), creation_context,
                                      model_creation_config.attn_config.model_config.quant_config.backend);

  decoder_layer_ = decoder_layer;
  tp_comm_ = std::make_shared<TpCommunicator>();
}

Status HyV3MtpLayer::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  RecordRequestSchedEvents(forwarding_context.GetBatchRequestSchedInfo(), forwarding_context.GetCurrentRank(),
                           forwarding_context.GetModelInput()->attn_dp_group_id_, "MTP", RequestEventPhase::Begin);
  auto& mtp_hidden_buffer = forwarding_context.GetForwardingBuffers()->GetNextStepHiddenBuffer();
  const auto& model_input = forwarding_context.GetModelInput();
  {
    CREATE_BUFFER_SCOPE(shared_buffer, forwarding_context.GetForwardingBuffers()->shared_buffer);

    // Embedding norm.
    enorm_->Forward(residual_buffer[0], residual_buffer[0]);

    // Gather last-token hidden.
    STATUS_CHECK_RETURN(
        static_cast<AssembleTokensHiddenLayer*>(gather_layer_.get())
            ->Forward(&mtp_hidden_buffer[0], model_input->GetEffectiveNextNGatherIdxTensor(), &shared_buffer[0]));

    // Hidden-state norm.
    hnorm_->Forward(shared_buffer[0], mtp_hidden_buffer[0]);

    // Concat [emb_norm, hidden_norm] along last dim.
    static_cast<ConcatLayer*>(concat_layer_.get())
        ->Forward(&residual_buffer[0], &mtp_hidden_buffer[0], &shared_buffer[0]);

    // Project 2*hidden -> hidden (column-parallel), then AllGather across TP.
    STATUS_CHECK_RETURN(eh_proj_->Forward(shared_buffer[0], residual_buffer));
    tp_comm_->AllGather(residual_buffer[0], mtp_hidden_buffer[0], forwarding_context);
  }
  const bool is_multi_token_forward = model_input->multi_token_request_num > 0;
  STATUS_CHECK_RETURN(decoder_layer_->Forward(residual_buffer, is_multi_token_forward, forwarding_context));

  RecordRequestSchedEvents(forwarding_context.GetBatchRequestSchedInfo(), forwarding_context.GetCurrentRank(),
                           forwarding_context.GetModelInput()->attn_dp_group_id_, "MTP", RequestEventPhase::End);
  return Status();
}

Status HyV3::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  return Status();
}

Status HyV3::CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) {
  auto& model_config = model_creation_config.attn_config.model_config;
  DataType weight_type = model_config.weight_data_type;
  first_k_dense_replace_ = model_config.moe_config.first_k_dense_replace;

  size_t max_token_num = creation_context.runtime_config.max_step_token_num;
  size_t moe_buffer_size = max_token_num * model_config.hidden_units;
  moe_buffer_ = creation_context.buffer_mgr_->CreateBufferTensor("moe_buffer_", {moe_buffer_size}, weight_type);

  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    const bool is_moe = (layer_idx >= first_k_dense_replace_);
    decoder_layers_[layer_idx] =
        std::make_shared<HyV3DecoderLayer>(layer_idx, is_moe, moe_buffer_, creation_context, model_creation_config);
  }
  return Status();
}

Status HyV3::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; layer_idx++) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  return Status();
}

HyV3Model::HyV3Model(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                     std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
                     std::shared_ptr<BaseModelCacheLayout> block_cache_layout)
    : CommonModel(model_config, runtime_config, rank, context) {
  ModelRunConfig model_run_config;
  hy_v3_.GetModelRunConfig(model_run_config, model_config);
  CommonModel::InitRunConfig(model_run_config, base_weight, block_cache_layout);
}

Status HyV3Model::CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) {
  STATUS_CHECK_RETURN(hy_v3_.CreateLayers(creation_context, model_creation_config));

  // Create MTP (NextN) layers when MTP is active.
  auto& model_config = model_creation_config.attn_config.model_config;
  if (pipeline_config_.lower_nextn_layer_idx >= static_cast<int>(model_config.num_layer)) {
    const int first_k_dense_replace = model_config.moe_config.first_k_dense_replace;
    const DataType weight_type = model_config.weight_data_type;
    const size_t max_token_num = creation_context.runtime_config.max_step_token_num;
    const size_t moe_buffer_size = max_token_num * model_config.hidden_units;
    TensorBuffer* mtp_moe_buffer =
        CommonModel::GetBufferManager()->CreateBufferTensor("mtp_moe_buffer", {moe_buffer_size}, weight_type);

    for (int layer_idx = pipeline_config_.lower_nextn_layer_idx; layer_idx <= pipeline_config_.upper_nextn_layer_idx;
         ++layer_idx) {
      const bool is_moe = layer_idx >= first_k_dense_replace;
      auto decoder_layer = std::make_shared<HyV3DecoderLayer>(layer_idx, is_moe, mtp_moe_buffer, creation_context,
                                                              model_creation_config);
      nextn_layers_[layer_idx] =
          std::make_shared<HyV3MtpLayer>(layer_idx, creation_context, model_creation_config, decoder_layer);
    }

    // hy_v3 MTP has a per-layer final_layernorm applied AFTER the decoder block and BEFORE the
    // shared_head (== main lm_head). Route the MTP logits path through this norm rather than the
    // main model.norm (which lm_head_prenorm_ uses for the main path). Matches vLLM
    // HYV3MTPDecoderLayer.final_layernorm (hy_v3_mtp.py:110,135).
    if (context_->global->IsSamplingNode()) {
      nextn_shared_head_norm_ = std::make_unique<Layernorm>(
          fmt::format("model.layers.{}.final_layernorm.weight", pipeline_config_.lower_nextn_layer_idx),
          model_config.layernorm_eps, creation_context);
    }

    // Allocate and inject the next-step hidden buffer for MTP (owned by the model).
    if (creation_context.runtime_config.mtp_step_num > 0) {
      next_step_hidden_tensors_ = {
          Tensor(MemoryLocation::LOCATION_DEVICE, weight_type,
                 {max_token_num * model_config.hidden_units * model_config.mhc_config.hc_mult})};
      forwarding_context_->GetForwardingBuffers()->SetExternalNextStepHiddenBuffer(&next_step_hidden_tensors_);
    }

    draft_model_ = std::make_unique<HyV3MtpDraftModel>(&nextn_layers_, nextn_shared_head_norm_.get(),
                                                       nextn_final_mhc_.get(), pipeline_config_.lower_nextn_layer_idx,
                                                       pipeline_config_.upper_nextn_layer_idx);
  }
  return Status();
}

Status HyV3Model::LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);

  if (run_mode == RunMode::kMain) {
    return hy_v3_.Forward(residual_buffer, forwarding_context);
  } else if (run_mode == RunMode::kNextN && draft_model_) {
    forwarding_context.SetIsForwardingLayers(false);  // Don't record ForwardingLayers event
    STATUS_CHECK_RETURN(draft_model_->ForwardLayer(residual_buffer, forwarding_context));
  }
  return Status();
}

}  // namespace ksana_llm
