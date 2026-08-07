/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/weight_instance.h"

#include <memory>

#include "utils/logger.h"
#include "utils/status.h"

#include "models/deepseek_v3/deepseek_v3_config.h"
#include "models/deepseek_v4/deepseek_v4_config.h"

namespace ksana_llm {

void WeightInstance::Load() {
  ModelConfigParser model_config_parser;
  std::shared_ptr<BaseModelConfig> model_config;
  Status status =
      model_config_parser.ParseModelConfig(model_config_.path, runtime_config_.parallel_basic_config, model_config);

  // Pass is_embedding_model from runtime ModelConfig to the weight loader's BaseModelConfig.
  // TODO(long-term): Consider passing is_embedding_model through ParseModelConfig parameters
  // instead of post-hoc assignment, to avoid this "patch-style" parameter passing.
  if (status.OK() && model_config) {
    model_config->is_embedding_model = model_config_.is_embedding_model;

    // Propagate fields that may have been changed by json_model_override_args from the flat
    // Environment::model_config_ into the freshly-parsed BaseModelConfig subclass.
    //
    // Background: Environment::ParseConfig applies json_model_override_args only to
    // Environment::model_config_ (flat ModelConfig); the ModelConfigParser::ParseModelConfig
    // call above independently re-parses raw config.json into a BaseModelConfig subclass and
    // is unaware of the override. Without this sync, e.g. DeepSeekV4Config::num_layer keeps the
    // original num_hidden_layers, and DeepSeekV4WeightLoader::FilterWeightNames /
    // ProcessModelWeights compute wrong MTP layer indices and silently skip MTP weights.
    //
    // TODO(long-term): replace this post-hoc patch with one of:
    //   (1) thread json_model_override_args through ModelConfigParser::ParseModelConfig so each
    //       derived parser applies the override itself; or
    //   (2) add a virtual ApplyOverride(const ModelConfig&) hook on BaseModelConfig that each
    //       derived config overrides to sync its own fields.
    // For now we follow the same patch-style as is_embedding_model above, covering only
    // DeepSeekV3 (and V4 by inheritance), where this bug first surfaced.
    if (auto* ds_cfg = dynamic_cast<DeepSeekV3Config*>(model_config.get())) {
      ds_cfg->num_layer = model_config_.num_layer;
      ds_cfg->num_nextn_predict_layers = model_config_.num_nextn_predict_layers;
      ds_cfg->hidden_units = model_config_.hidden_units;
      ds_cfg->head_num = model_config_.head_num;
      ds_cfg->num_key_value_heads = model_config_.num_key_value_heads;
      ds_cfg->inter_size = model_config_.inter_size;
      ds_cfg->size_per_head = model_config_.size_per_head;
      ds_cfg->vocab_size = model_config_.vocab_size;
      ds_cfg->rotary_embedding = model_config_.rotary_embedding;
      ds_cfg->max_position_embeddings = model_config_.max_position_embeddings;
      ds_cfg->rope_theta = model_config_.rope_theta;
      ds_cfg->layernorm_eps = model_config_.layernorm_eps;
    }

    // dsv4 also carries compress_ratios, which Environment has already realigned to
    // num_layer + num_nextn_predict_layers via RealignDsv4CompressRatiosAfterOverride.
    // Copy it directly to avoid re-implementing the realignment here. No current reader
    // on this BaseModelConfig path consumes compress_ratios, but keeping it in sync
    // prevents future code from tripping on the same bug.
    if (auto* dsv4_cfg = dynamic_cast<DeepSeekV4Config*>(model_config.get())) {
      dsv4_cfg->dsv4_attn_config.compress_ratios = model_config_.dsv4_attn_config.compress_ratios;
      // Runtime mtp_step_num overrides dspark_config
      dsv4_cfg->dspark_config.dspark_block_size = model_config_.dspark_config.dspark_block_size;
      dsv4_cfg->dspark_config.dspark_target_layer_ids = model_config_.dspark_config.dspark_target_layer_ids;
    }

    // TODO(non-DS-models): if any non-DeepSeek model (Llama / Qwen / Qwen2Moe / Qwen3Moe /
    // Mixtral / Glm4Moe / ArcHunyuanVideo / HunyuanTurbo / Internlm2 / Internlmxcomposer2 /
    // Chatglm / BgeRerankerMinicpm etc.) starts using json_model_override_args to change
    // architectural fields (num_layer / hidden_units / ...), it will hit the same bug.
    // Either add another dynamic_cast branch here, or migrate to one of the long-term
    // solutions above.
  }

  if (status.OK()) {
    KLLM_LOG_INFO << "Start to load model weights";
    ModelWeightLoader weight_loader(Singleton<Environment>::GetInstance(), context_);

    std::shared_ptr<ModelWeight> dev_weight = std::make_shared<ModelWeight>();
    status = weight_loader.LoadWeight(model_config, dev_weight);
    STATUS_CHECK_FAILURE(status);
    weight_ = dev_weight;
  }
}

}  // namespace ksana_llm
