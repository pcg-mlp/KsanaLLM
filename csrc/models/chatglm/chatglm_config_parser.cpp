/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/chatglm/chatglm_config_parser.h"

#include <memory>

#include "configure/json_config_utils.h"
#include "models/chatglm/chatglm_model_config.h"

namespace ksana_llm {

ChatglmConfigParser::ChatglmConfigParser() {}

ChatglmConfigParser::~ChatglmConfigParser() {}

Status ChatglmConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                             const ParallelismBasicConfig &parallel_basic_config,
                                             const std::string &model_dir,
                                             std::shared_ptr<BaseModelConfig> &model_config) {
  std::shared_ptr<ChatglmModelConfig> chatglm_config = std::make_shared<ChatglmModelConfig>();
  model_config = chatglm_config;

  chatglm_config->weight_data_type = GetModelDataType(config_json);

  // Parse ChatGLM specific config fields
  chatglm_config->head_num = config_json.at("num_attention_heads");
  // ChatGLM uses multi_query_group_num for GQA, fallback to head_num for MHA
  chatglm_config->num_key_value_heads =
      config_json.value("multi_query_group_num", static_cast<size_t>(chatglm_config->head_num));
  chatglm_config->inter_size = config_json.at("ffn_hidden_size");
  chatglm_config->vocab_size = config_json.value("vocab_size", 65024);
  // GLM-4 uses padded_vocab_size
  chatglm_config->vocab_size = config_json.value("padded_vocab_size", chatglm_config->vocab_size);
  chatglm_config->num_layer = config_json.value("num_layers", 28);
  chatglm_config->hidden_units = config_json.at("hidden_size");

  // Prefer explicit rope_theta if available, otherwise compute from rope_ratio
  if (config_json.contains("rope_theta")) {
    chatglm_config->rope_theta = config_json.at("rope_theta").get<float>();
  } else {
    float rope_ratio = config_json.value("rope_ratio", 1.0f);
    chatglm_config->rope_theta = rope_ratio * 10000.0f;
  }

  chatglm_config->layernorm_eps = config_json.value("layernorm_epsilon", 1e-5f);
  chatglm_config->start_id = config_json.value("bos_token_id", 1);

  // GLM-4 may have multiple end tokens (array)
  if (config_json.contains("eos_token_id") && config_json["eos_token_id"].is_array()) {
    chatglm_config->end_ids = config_json["eos_token_id"].get<std::vector<uint32_t>>();
  } else {
    chatglm_config->end_ids =
        std::vector<uint32_t>{static_cast<uint32_t>(config_json.value("eos_token_id", 2))};
  }

  chatglm_config->pad_id = config_json.value("pad_token_id", 0);
  // ChatGLM uses seq_length instead of max_position_embeddings
  chatglm_config->max_position_embeddings = config_json.value("seq_length", 32768);
  chatglm_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);
  chatglm_config->is_visual = config_json.contains("visual");

  // Calculate size_per_head
  chatglm_config->size_per_head = chatglm_config->hidden_units / chatglm_config->head_num;
  // ChatGLM uses half of size_per_head for rotary embedding
  chatglm_config->rotary_embedding = chatglm_config->size_per_head / 2;

  return Status();
}

}  // namespace ksana_llm
