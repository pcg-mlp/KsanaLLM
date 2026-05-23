/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/qwen/qwen_model.h"

#include <memory>
#include <string>
#include <vector>

#include "utils/logger.h"

namespace ksana_llm {

Status Qwen::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  return Status();
}

Status Qwen::CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) {
  bool is_neox = true;
  bool add_qkv_bias = model_creation_config.attn_config.model_config.enable_add_qkv_bias;
  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    decoder_layers_[layer_idx] =
        std::make_shared<SimpleDecoderLayer>(layer_idx, is_neox, add_qkv_bias, creation_context, model_creation_config);
  }
  if (creation_context.model_config.num_deepstack_layers > 0) {
    deepstack_add_ = std::make_shared<Add>(creation_context);
  }
  return Status();
}

Status Qwen::InjectDeepstack(std::vector<Tensor>& residual_buffer, const ModelInput::DeepstackData& deepstack,
                             int layer_idx) {
  auto it = deepstack.layer_to_deepstack_idx.find(layer_idx);
  if (it == deepstack.layer_to_deepstack_idx.end()) return Status();

  const Tensor& layer_embed = deepstack.layer_embeds[it->second];
  size_t hidden_size = layer_embed.shape[1];

  for (const auto& seg : deepstack.segments) {
    Tensor residual_view = residual_buffer[0].GetView({static_cast<size_t>(seg.length), hidden_size},
                                                      static_cast<size_t>(seg.batch_pos) * hidden_size);
    Tensor embed_view = layer_embed.GetView({static_cast<size_t>(seg.length), hidden_size},
                                            static_cast<size_t>(seg.embed_offset) * hidden_size);
    std::vector<Tensor> output = {residual_view};
    STATUS_CHECK_RETURN(deepstack_add_->Forward(residual_view, embed_view, output));
  }
  return Status();
}

Status Qwen::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  auto& deepstack = forwarding_context.GetModelInput()->deepstack_data;

  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));

    if (is_multi_token_forward && deepstack.active) {
      STATUS_CHECK_RETURN(InjectDeepstack(residual_buffer, deepstack, layer_idx));
    }
  }

  return Status();
}

/* **************************************
 * QwenModel
 */
QwenModel::QwenModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                     std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
                     std::shared_ptr<BaseModelCacheLayout> block_cache_layout)
    : CommonModel(model_config, runtime_config, rank, context) {
  ModelRunConfig model_run_config;
  qwen_.GetModelRunConfig(model_run_config, model_config);
  CommonModel::InitRunConfig(model_run_config, base_weight, block_cache_layout);
}

Status QwenModel::CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) {
  return qwen_.CreateLayers(creation_context, model_creation_config);
}

Status QwenModel::LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferTensors(forwarding_context);
  STATUS_CHECK_RETURN(qwen_.Forward(residual_buffer, forwarding_context));

  return Status();
}

}  // namespace ksana_llm
