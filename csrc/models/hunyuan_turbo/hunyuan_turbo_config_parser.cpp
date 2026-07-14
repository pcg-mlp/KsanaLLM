/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/hunyuan_turbo/hunyuan_turbo_config_parser.h"

#include "configure/json_config_utils.h"

namespace ksana_llm {

HunyuanTurboConfigParser::HunyuanTurboConfigParser() {}

HunyuanTurboConfigParser::~HunyuanTurboConfigParser() {}

Status HunyuanTurboConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                                  const ParallelismBasicConfig &parallel_basic_config,
                                                  const std::string &model_dir,
                                                  std::shared_ptr<BaseModelConfig> &model_config) {
  // TODO(ethanyczeng): 还需要根据HunYuanTurbo中的模型结构进行进一步的适配
  std::shared_ptr<HunyuanTurboConfig> hunyuan_turbo_config = std::make_shared<HunyuanTurboConfig>();
  model_config = hunyuan_turbo_config;

  hunyuan_turbo_config->weight_data_type = GetModelDataType(config_json);

  hunyuan_turbo_config->head_num = config_json.value("num_attention_heads", 32);
  hunyuan_turbo_config->num_key_value_heads = config_json.value("num_key_value_heads", 32);
  hunyuan_turbo_config->inter_size = config_json.value("intermediate_size", 11008);
  hunyuan_turbo_config->vocab_size = config_json.value("vocab_size", 32000);
  hunyuan_turbo_config->num_layer = config_json.value("num_hidden_layers", 32);
  hunyuan_turbo_config->hidden_units = config_json.value("hidden_size", 4096);
  hunyuan_turbo_config->rope_theta = config_json.value("rope_theta", 10000.0f);
  hunyuan_turbo_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-6);
  hunyuan_turbo_config->start_id = config_json.value("bos_token_id", 1);
  hunyuan_turbo_config->end_id = config_json.value("eos_token_id", 2);
  hunyuan_turbo_config->pad_id = config_json.value("pad_token_id", 0);
  hunyuan_turbo_config->max_position_embeddings = config_json.value("max_position_embeddings", 2048);

  if (!config_json.contains("tie_word_embeddings")) {
    hunyuan_turbo_config->exist_tie_embeddings_param = false;
  }
  hunyuan_turbo_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);

  size_t size_per_head =
      config_json.value("head_dim", hunyuan_turbo_config->hidden_units / hunyuan_turbo_config->head_num);
  hunyuan_turbo_config->size_per_head = size_per_head;

  // Parse MoE config
  ParseMoeConfig(config_json, hunyuan_turbo_config);

  return Status();
}

template <typename T>
inline T ExtractPossiblyRepeatedArray(const nlohmann::json &config_json, const std::string &key, T default_value) {
  // This function extracts a value from a JSON object, which may be an array or a single value.
  if (!config_json.contains(key)) {
    return default_value;
  }
  if (config_json[key].is_array()) {
    auto values = config_json[key].get<std::vector<T>>();
    if (values.empty()) {
      return default_value;
    }
    bool all_same = std::adjacent_find(values.begin(), values.end(), std::not_equal_to<>()) == values.end();
    if (!all_same) {
      // TODO(ethanyczeng): if values are not same, we need to do other operation.
      KLLM_LOG_WARNING << "Found different values in " << key << ", using first value: " << values[0];
    }
    return values[0];
  }
  return config_json.value(key, default_value);
}

Status HunyuanTurboConfigParser::ParseMoeConfig(const nlohmann::json &config_json,
                                                std::shared_ptr<HunyuanTurboConfig> hunyuan_turbo_config) {
  hunyuan_turbo_config->moe_config.num_experts = config_json.value("num_experts", 1);
  // Handle possibly array-based values with single value check
  hunyuan_turbo_config->moe_config.experts_topk = ExtractPossiblyRepeatedArray(config_json, "moe_topk", 0);
  hunyuan_turbo_config->moe_config.num_shared_experts =
      ExtractPossiblyRepeatedArray(config_json, "num_shared_expert", 0);

  // For arrays that might contain different values but you still need just one representative value
  int moe_inter_size =
      ExtractPossiblyRepeatedArray(config_json, "moe_intermediate_size", hunyuan_turbo_config->inter_size);
  hunyuan_turbo_config->moe_config.moe_inter_size = moe_inter_size;

  if (hunyuan_turbo_config->moe_config.num_experts > 1 && config_json["moe_topk"].is_number()) {
    hunyuan_turbo_config->is_moe = true;
    hunyuan_turbo_config->moe_config.moe_inter_size = hunyuan_turbo_config->inter_size;
    hunyuan_turbo_config->moe_config.experts_topk = config_json.value("moe_topk", 1);
  } else if (hunyuan_turbo_config->moe_config.num_experts == 1) {
    hunyuan_turbo_config->is_moe = false;
  }
  // hunyuan_turbo_config->moe_config.num_shared_experts = config_json.value("num_shared_expert", 0);
  KLLM_LOG_DEBUG << fmt::format("num_shared_expert: {}", hunyuan_turbo_config->moe_config.num_shared_experts);
  if (hunyuan_turbo_config->moe_config.num_shared_experts > 0 && config_json["num_shared_expert"].is_number()) {
    hunyuan_turbo_config->has_shared_experts = true;
    hunyuan_turbo_config->moe_config.shared_expert_inter_size = hunyuan_turbo_config->inter_size;
  }
  hunyuan_turbo_config->cla_share_factor = config_json.value("cla_share_factor", 0);
  if (hunyuan_turbo_config->is_moe && hunyuan_turbo_config->cla_share_factor != 0)
    hunyuan_turbo_config->load_bias =
        false;  // The current Hunyuan-large model has saved the bias weights, but does not utilize any bias weights
                // during inference. We will continue to monitor this aspect in the future.
  hunyuan_turbo_config->use_qk_norm = config_json.value("use_qk_norm", true);
  hunyuan_turbo_config->mlp_bias = config_json.value("mlp_bias", false);
  hunyuan_turbo_config->use_cla = config_json.value("use_cla", false);
  KLLM_LOG_INFO << fmt::format("using moe model, num_experts: {}", hunyuan_turbo_config->moe_config.num_experts);

  return Status();
}

}  // namespace ksana_llm
