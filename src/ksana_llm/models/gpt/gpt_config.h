/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/environment.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

inline void PrepareFairseqTransformerAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  // Refer to
  // https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_config.py
  model_config.head_num = config_json.value("n_head", 8);
  model_config.num_key_value_heads = model_config.head_num;
  model_config.vocab_size = config_json.value("vocab_size", 7000);
  model_config.num_layer = config_json.value("n_layer", 6);
  model_config.hidden_units = config_json.value("n_embd", 512);
  model_config.inter_size = config_json.value("n_inner", 4 * model_config.hidden_units);
  model_config.layernorm_eps = config_json.value("layer_norm_epsilon", 1e-5);  // torch.nn uses 1e-5 by default
  model_config.start_id = config_json.value("bos_token_id", 0);
  model_config.end_ids = std::vector<uint32_t>{static_cast<uint32_t>(config_json.value("eos_token_id", 2))};
  model_config.pad_id = config_json.value("pad_token_id", 1);
  model_config.max_position_embeddings = config_json.value("n_positions", 512);
  model_config.activation_function = config_json.value("afn", "relu");
  // Tie the weights of the token_embedding and the lm_head.
  model_config.tie_word_embeddings = true;

  size_t size_per_head = model_config.hidden_units / model_config.head_num;
  model_config.size_per_head = size_per_head;
  model_config.rotary_embedding = size_per_head;
}

inline void PrepareGPT1Attributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  // Refer to
  // https://github.com/huggingface/transformers/blob/main/src/transformers/models/openai/configuration_openai.py
  model_config.head_num = config_json.value("n_head", 12);
  model_config.num_key_value_heads = model_config.head_num;
  model_config.vocab_size = config_json.value("vocab_size", 40478);
  model_config.num_layer = config_json.value("n_layer", 12);
  model_config.hidden_units = config_json.value("n_embd", 768);
  model_config.inter_size = config_json.value("n_inner", 4 * model_config.hidden_units);
  model_config.layernorm_eps = config_json.value("layer_norm_epsilon", 1e-5);
  // Note: GPT-1 does not have bos, eos and pad tokens by default.
  model_config.start_id = config_json.value("bos_token_id", 0);
  if (config_json.contains("eos_token_id")) {
    model_config.end_ids = std::vector<uint32_t>{static_cast<uint32_t>(config_json.at("eos_token_id"))};
  }
  model_config.pad_id = config_json.value("pad_token_id", 0);
  model_config.max_position_embeddings = config_json.value("n_positions", 512);
  model_config.activation_function = config_json.value("afn", "gelu");
  // Tie the weights of the token_embedding and the lm_head.
  model_config.tie_word_embeddings = true;

  size_t size_per_head = model_config.hidden_units / model_config.head_num;
  model_config.size_per_head = size_per_head;
  model_config.rotary_embedding = size_per_head;
}

inline void PrepareGPT2Attributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  // Refer to
  // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py
  model_config.head_num = config_json.value("n_head", 12);
  model_config.num_key_value_heads = model_config.head_num;
  model_config.vocab_size = config_json.value("vocab_size", 50257);
  model_config.num_layer = config_json.value("n_layer", 12);
  model_config.hidden_units = config_json.value("n_embd", 768);
  model_config.inter_size = config_json.value("n_inner", 4 * model_config.hidden_units);
  model_config.layernorm_eps = config_json.value("layer_norm_epsilon", 1e-5);
  model_config.start_id = config_json.value("bos_token_id", 50256);
  model_config.end_ids = std::vector<uint32_t>{config_json.value("eos_token_id", 50256)};
  model_config.pad_id = config_json.value("pad_token_id", 0);
  model_config.max_position_embeddings = config_json.value("n_positions", 1024);
  model_config.activation_function = config_json.value("activation_function", "gelu_new");
  // Tie the weights of the token_embedding and the lm_head.
  model_config.tie_word_embeddings = true;

  size_t size_per_head = model_config.hidden_units / model_config.head_num;
  model_config.size_per_head = size_per_head;
  model_config.rotary_embedding = size_per_head;
}

}  // namespace ksana_llm
