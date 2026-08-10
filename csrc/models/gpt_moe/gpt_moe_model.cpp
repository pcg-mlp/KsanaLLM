/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/gpt_moe/gpt_moe_model.h"

namespace ksana_llm {

GptMoeDecoderLayer::GptMoeDecoderLayer(int layer_idx, LayerCreationContext& creation_context,
                                       ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx) {
  const std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  // Common blocks
  adds_ = std::make_unique<Add>(creation_context);
  tp_comm_ = std::make_shared<TpCommunicator>();

  // LayerNorm without bias
  input_layernorms_ =
      std::make_unique<Layernorm>(layer_prefix + ".input_layernorm.weight",
                                  model_creation_config.layernorm_config.layernorm_eps, creation_context, "layernorm");
  post_attention_layernorms_ =
      std::make_unique<Layernorm>(layer_prefix + ".post_attention_layernorm.weight",
                                  model_creation_config.layernorm_config.layernorm_eps, creation_context, "layernorm");

  // Mha blocks
  const bool is_neox = true;
  const bool add_qkv_bias = false;
  const bool use_qk_norm = false;
  mha_ = std::make_unique<MultiHeadAttention>(layer_idx, is_neox, add_qkv_bias, use_qk_norm, creation_context,
                                              model_creation_config);
  // MoE blocks
  expert_gates_ = std::make_unique<Linear>(layer_prefix + ".mlp.gate.weight", creation_context,
                                           model_creation_config.attn_config.model_config.quant_config.backend);
  moes_ = std::make_unique<MoE>(layer_idx, layer_prefix + ".mlp.experts.gate_proj.weight",
                                layer_prefix + ".mlp.experts.down_proj.weight", creation_context,
                                MoeScaleNormMode::RE_NORM);
}

Status GptMoeDecoderLayer::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                                   ForwardingContext& forwarding_context) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);

  input_layernorms_->Forward(residual_buffer[0], hidden_buffer_tensors_0[0]);

  // MultiHeadAttention
  STATUS_CHECK_RETURN(
      mha_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Attn residual add
  STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));

  // Pre layernorm uses layernorm input for residual connection.
  post_attention_layernorms_->Forward(residual_buffer[0], hidden_buffer_tensors_0[0]);

  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
  STATUS_CHECK_RETURN(expert_gates_->Forward(hidden_buffer_tensors_0[0], hidden_buffer_tensors_1[0]));

  // Moe
  STATUS_CHECK_RETURN(moes_->Forward(hidden_buffer_tensors_0[0], hidden_buffer_tensors_1[0],
                                     /*unused*/ reduce_buffer_tensors[0], reduce_buffer_tensors));

  if (forwarding_context.GetModelCommunicator()) {
    tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

    // Moe residual add
    STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));
  } else {
    // Moe residual add
    STATUS_CHECK_RETURN(adds_->Forward(reduce_buffer_tensors[0], residual_buffer[0], residual_buffer));
  }
  return Status();
}

Status GptMoe::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::LEARNED_ABSOLUTE;
  model_run_config.norm_type = NormType::LAYERNORM;
  model_run_config.emb_lookup_use_rotary_embedding_pos = true;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  return Status();
}

Status GptMoe::CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) {
  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    decoder_layers_[layer_idx] =
        std::make_unique<GptMoeDecoderLayer>(layer_idx, creation_context, model_creation_config);
  }
  return Status();
}

Status GptMoe::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  return Status();
}

/**********************************************************
 * GptMoeModel
 ***********************************************************/
GptMoeModel::GptMoeModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                         std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
                         std::shared_ptr<BaseModelCacheLayout> block_cache_layout)
    : CommonModel(model_config, runtime_config, rank, context) {
  ModelRunConfig model_run_config;
  gptmoe_.GetModelRunConfig(model_run_config, model_config);
  CommonModel::InitRunConfig(model_run_config, base_weight, block_cache_layout);
}

Status GptMoeModel::CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) {
  return gptmoe_.CreateLayers(creation_context, model_creation_config);
}

Status GptMoeModel::LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);
  return gptmoe_.Forward(residual_buffer, forwarding_context);
}

}  // namespace ksana_llm
