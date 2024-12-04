/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/environment.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

inline void PrepareChatglmAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  model_config.head_num = config_json.at("num_attention_heads");
  model_config.num_key_value_heads = config_json.value("multi_query_group_num", model_config.head_num);
  model_config.inter_size = config_json.at("ffn_hidden_size");
  model_config.vocab_size = config_json.value("vocab_size", 65024);
  model_config.vocab_size = config_json.value("padded_vocab_size", model_config.vocab_size);  // for glm4 config
  model_config.num_layer = config_json.value("num_layers", 28);
  model_config.hidden_units = config_json.at("hidden_size");
  model_config.rope_theta = config_json.value("rope_ratio", 1.0f);
  model_config.rope_theta = model_config.rope_theta * 10000;
  model_config.layernorm_eps = config_json.value("layernorm_epsilon", 1e-5);
  model_config.start_id = config_json.value("bos_token_id", 1);
  // for glm4 config
  if (config_json.contains("eos_token_id") && config_json["eos_token_id"].is_array()) {
    model_config.end_ids = config_json["eos_token_id"].get<std::vector<uint32_t>>();
  } else {
    model_config.end_ids = std::vector<uint32_t>{static_cast<uint32_t>(config_json.value("eos_token_id", 2))};
  }
  model_config.pad_id = config_json.value("pad_token_id", 0);
  model_config.max_position_embeddings = config_json.value("seq_length", 32768);
  model_config.tie_word_embeddings = config_json.value("tie_word_embeddings", false);
  model_config.is_visual = config_json.contains("visual");

  size_t size_per_head = model_config.hidden_units / model_config.head_num;
  model_config.size_per_head = size_per_head;
  model_config.rotary_embedding = size_per_head / 2;
}

}  // namespace ksana_llm
