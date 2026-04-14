/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/glm4_moe/glm4_moe_model.h"

#include "device/device_utils.h"
#include "layers/assemble_tokens_hidden_layer.h"
#include "layers/concat_layer.h"
#include "layers/emb_lookup_layer.h"
#include "profiler/profile_event.h"
#include "profiler/sched_event_tracer.h"

namespace ksana_llm {

Glm4MoeDecoderLayer::Glm4MoeDecoderLayer(int layer_idx, bool is_moe, TensorBuffer* moe_buffer,
                                         LayerCreationContext& creation_context,
                                         ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx), is_moe_(is_moe), moe_buffer_(moe_buffer) {
  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  add_ = std::make_shared<Add>(creation_context);
  tp_comm_ = std::make_shared<TpCommunicator>();

  input_layernorm_ = std::make_shared<Layernorm>(
      layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);

  post_attention_layernorm_ =
      std::make_shared<Layernorm>(layer_prefix + ".post_attention_layernorm.weight",
                                  model_creation_config.layernorm_config.layernorm_eps, creation_context);

  // MultiHeadAttention with QK Norm
  bool is_neox = true;
  bool add_qkv_bias = true;
  bool use_qk_norm = true;
  mha_ = std::make_shared<MultiHeadAttention>(layer_idx, is_neox, add_qkv_bias, use_qk_norm, creation_context,
                                              model_creation_config);

  if (is_moe) {
    expert_gate_ = std::make_shared<Linear>(layer_prefix + ".mlp.gate.weight", creation_context,
                                            model_creation_config.attn_config.model_config.quant_config.backend);

    // MoE with e_score_correction_bias
    if (model_creation_config.attn_config.model_config.moe_config.use_e_score_correction_bias) {
      moe_ = std::make_shared<MoE>(
          layer_idx, layer_prefix + ".mlp.experts.up_gate_proj.weight", layer_prefix + ".mlp.experts.down_proj.weight",
          layer_prefix + ".mlp.gate.e_score_correction_bias", creation_context, MoeScaleNormMode::NO_NORM);
    } else {
      moe_ = std::make_shared<MoE>(layer_idx, layer_prefix + ".mlp.experts.up_gate_proj.weight",
                                   layer_prefix + ".mlp.experts.down_proj.weight", creation_context,
                                   MoeScaleNormMode::NO_NORM);
    }

    // Shared expert (single FFN with larger intermediate size)
    shared_mlp_ = std::make_shared<TwoLayeredFFN>(layer_idx, creation_context, model_creation_config,
                                                  ".mlp.shared_expert.{}.weight");
  } else {
    // Dense FFN layer
    mlp_ = std::make_shared<TwoLayeredFFN>(layer_idx, creation_context, model_creation_config);
  }
}

Status Glm4MoeDecoderLayer::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                                    ForwardingContext& forwarding_context) {
  KLLM_LOG_DEBUG << fmt::format("Layer {}: Forward START", layer_idx_);
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);

  // Pre-Norm: LayerNorm before attention
  input_layernorm_->Forward(residual_buffer, hidden_buffer_tensors_0);

  // MultiHeadAttention (with QK Norm handled inside MHA)
  STATUS_CHECK_RETURN(
      mha_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));

  // AllReduce for attention output
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Residual connection after attention
  STATUS_CHECK_RETURN(add_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));

  // Pre-Norm: LayerNorm before MLP/MoE
  post_attention_layernorm_->Forward(residual_buffer, hidden_buffer_tensors_0);

  // MLP or MoE
  STATUS_CHECK_RETURN(
      ForwardMlp(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));

  // AllReduce for MLP/MoE output
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Residual connection after MLP/MoE
  STATUS_CHECK_RETURN(add_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));

  KLLM_LOG_DEBUG << fmt::format("Layer {}: Forward END", layer_idx_);
  return Status();
}

Status Glm4MoeDecoderLayer::ForwardMlp(std::vector<Tensor>& hidden_buffer_tensors_0,
                                       std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                       ForwardingContext& forwarding_context) {
  if (!is_moe_) {
    // Dense FFN layer
    mlp_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context);
  } else {
    // MoE layer
    CREATE_BUFFER_SCOPE(moe_buffer_tensors, moe_buffer_);
    auto& gated_buffer_ = reduce_buffer_tensors;

    // Expert gating: compute router logits
    STATUS_CHECK_RETURN(expert_gate_->Forward(hidden_buffer_tensors_0, gated_buffer_));

    // Routed experts: process hidden_buffer_tensors_0 with routing
    STATUS_CHECK_RETURN(
        moe_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0], reduce_buffer_tensors[0], moe_buffer_tensors));

    // Save routed expert output to reduce_buffer_tensors[0] before shared expert overwrites moe_buffer_tensors[0]
    size_t routed_output_size = moe_buffer_tensors[0].GetTotalBytes();
    MemcpyAsync(reduce_buffer_tensors[0].GetPtr<void>(), moe_buffer_tensors[0].GetPtr<void>(), routed_output_size,
                MEMCPY_DEVICE_TO_DEVICE, forwarding_context.GetContext()->device->GetPrimaryComputeStream());
    reduce_buffer_tensors[0].shape = moe_buffer_tensors[0].shape;
    reduce_buffer_tensors[0].dtype = moe_buffer_tensors[0].dtype;

    // Shared expert: process the SAME input hidden_buffer_tensors_0
    // IMPORTANT: TwoLayeredFFN::Forward output location depends on TP:
    //   - TP > 1 (GetModelCommunicator() != nullptr): output in reduce_buffer_tensors (= moe_buffer_tensors here)
    //   - TP = 1 (GetModelCommunicator() == nullptr): output in hidden_buffer_tensors_0 (via internal swap)
    STATUS_CHECK_RETURN(
        shared_mlp_->Forward(hidden_buffer_tensors_0, moe_buffer_tensors, is_multi_token_forward, forwarding_context));

    // Determine where shared expert output actually is
    bool has_model_communicator = (forwarding_context.GetModelCommunicator() != nullptr);
    auto& shared_expert_output = has_model_communicator ? moe_buffer_tensors[0] : hidden_buffer_tensors_0[0];

    // Combine routed and shared expert outputs
    if (has_model_communicator) {
      STATUS_CHECK_RETURN(add_->Forward(reduce_buffer_tensors[0], shared_expert_output, reduce_buffer_tensors));
    } else {
      STATUS_CHECK_RETURN(add_->Forward(reduce_buffer_tensors[0], shared_expert_output, hidden_buffer_tensors_0));
    }
  }
  return Status();
}

/**********************************************************
 * Glm4MoeMtpLayer
 ***********************************************************/

Glm4MoeMtpLayer::Glm4MoeMtpLayer(int layer_idx, TensorBuffer* moe_buffer, LayerCreationContext& creation_context,
                                 ModelCreationConfig& model_creation_config,
                                 std::shared_ptr<Glm4MoeDecoderLayer> decoder_layer) {
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

Status Glm4MoeMtpLayer::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
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
  STATUS_CHECK_RETURN(decoder_layer_->Forward(residual_buffer, is_multi_token_forward, forwarding_context));

  RecordRequestSchedEvents(forwarding_context.GetBatchRequestSchedInfo(), forwarding_context.GetCurrentRank(),
                           forwarding_context.GetModelInput()->attn_dp_group_id_, "MTP", RequestEventPhase::End);
  return Status();
}

/**********************************************************
 * Glm4Moe
 ***********************************************************/

Status Glm4Moe::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  return Status();
}

Status Glm4Moe::CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) {
  auto& model_config = model_creation_config.attn_config.model_config;
  DataType weight_type = model_config.weight_data_type;

  // Get first_k_dense_replace from model config
  first_k_dense_replace_ = model_config.moe_config.first_k_dense_replace;

  // Create MoE buffer
  size_t max_token_num = creation_context.runtime_config.max_step_token_num;
  size_t moe_buffer_size = max_token_num * model_config.hidden_units;
  moe_buffer_ = creation_context.buffer_mgr_->CreateBufferTensor("moe_buffer_", {moe_buffer_size}, weight_type);

  // Create decoder layers
  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    bool is_moe = (layer_idx >= first_k_dense_replace_);
    decoder_layers_[layer_idx] =
        std::make_shared<Glm4MoeDecoderLayer>(layer_idx, is_moe, moe_buffer_, creation_context, model_creation_config);
  }
  return Status();
}

Status Glm4Moe::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; layer_idx++) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  return Status();
}

Glm4MoeModel::Glm4MoeModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                           std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight)
    : CommonModel(model_config, runtime_config, rank, context) {
  ModelRunConfig model_run_config;
  glm4moe_.GetModelRunConfig(model_run_config, model_config);

  CommonModel::InitRunConfig(model_run_config, base_weight);
}

Status Glm4MoeModel::CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) {
  Status status = glm4moe_.CreateLayers(creation_context, model_creation_config);
  STATUS_CHECK_RETURN(status);

  // Create MTP layers if configured
  auto& model_config = model_creation_config.attn_config.model_config;
  if (pipeline_config_.lower_nextn_layer_idx >= static_cast<int>(model_config.num_layer)) {
    int first_k_dense_replace = model_config.moe_config.first_k_dense_replace;
    DataType weight_type = model_config.weight_data_type;
    size_t max_token_num = creation_context.runtime_config.max_step_token_num;
    size_t moe_buffer_size = max_token_num * model_config.hidden_units;
    // Create separate MoE buffer for MTP decoder layers
    TensorBuffer* moe_buffer =
        CommonModel::GetBufferManager()->CreateBufferTensor("mtp_moe_buffer", {moe_buffer_size}, weight_type);

    for (int layer_idx = pipeline_config_.lower_nextn_layer_idx; layer_idx <= pipeline_config_.upper_nextn_layer_idx;
         ++layer_idx) {
      const bool is_moe = layer_idx >= first_k_dense_replace;
      auto decoder_layer =
          std::make_shared<Glm4MoeDecoderLayer>(layer_idx, is_moe, moe_buffer, creation_context, model_creation_config);
      nextn_layers_[layer_idx] = std::make_shared<Glm4MoeMtpLayer>(layer_idx, moe_buffer, creation_context,
                                                                   model_creation_config, decoder_layer);
    }
    nextn_layer_idx_ = pipeline_config_.lower_nextn_layer_idx;
  }
  return Status();
}

Status Glm4MoeModel::LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);

  if (run_mode == RunMode::kMain) {
    return glm4moe_.Forward(residual_buffer, forwarding_context);
  } else if (run_mode == RunMode::kNextN && !nextn_layers_.empty()) {
    forwarding_context.SetIsForwardingLayers(false);
    for (int layer_idx = nextn_layer_idx_; layer_idx < nextn_layer_idx_ + static_cast<int>(kNextnDecoderLayerNum);
         ++layer_idx) {
      STATUS_CHECK_RETURN(nextn_layers_[layer_idx]->Forward(residual_buffer, forwarding_context));
    }

    nextn_layer_idx_ += kNextnDecoderLayerNum;
    if (nextn_layer_idx_ >=
        pipeline_config_.lower_nextn_layer_idx + static_cast<int>(model_config_.num_nextn_predict_layers)) {
      nextn_layer_idx_ = pipeline_config_.lower_nextn_layer_idx;
    }
  }
  return Status();
}

}  // namespace ksana_llm
