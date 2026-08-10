/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/model_config.h"
#include "device/device_types.h"
#include "models/base/base_model_config.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

inline void PrepareGPTMoEAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  model_config.head_num = config_json.value("n_head", 8);
  model_config.num_key_value_heads = model_config.head_num;
  model_config.vocab_size = config_json.value("vocab_size", 6999);
  model_config.num_layer = config_json.value("n_layer", 6);
  model_config.hidden_units = config_json.value("n_embd", 512);
  model_config.inter_size = config_json.value("n_inner", 4 * model_config.hidden_units);
  model_config.layernorm_eps = config_json.value("layer_norm_epsilon", 1e-5);  // torch.nn uses 1e-5 by default
  model_config.start_id = config_json.value("bos_token_id", 0);
  model_config.end_ids = std::vector<uint32_t>{static_cast<uint32_t>(config_json.value("eos_token_id", 2))};
  model_config.pad_id = config_json.value("pad_token_id", 1);
  model_config.max_position_embeddings = config_json.value("n_positions", 128);
  model_config.activation_function = config_json.value("afn", "gelu");
  // Tie the weights of the token_embedding and the lm_head.
  model_config.tie_word_embeddings = true;

  size_t size_per_head = model_config.hidden_units / model_config.head_num;
  model_config.size_per_head = size_per_head;
  model_config.rotary_embedding = size_per_head;

  model_config.is_moe = true;
  model_config.moe_config.num_experts = config_json.value("num_experts", 8);
  model_config.moe_config.moe_inter_size = config_json.value("moe_intermediate_size", 2 * model_config.hidden_units);
  model_config.moe_config.experts_topk = config_json.value("num_experts_per_tok", 2);
}

// Model config for gpt_moe architecture.
struct GptMoeConfig : public BaseModelConfig {
  virtual ~GptMoeConfig() = default;

  DataType weight_data_type;

  size_t head_num;
  uint32_t size_per_head;
  uint32_t inter_size;
  uint32_t hidden_units;
  uint32_t num_layer;
  uint32_t rotary_embedding;
  float layernorm_eps;
  uint32_t vocab_size;
  uint32_t start_id;
  std::vector<uint32_t> end_ids;
  uint32_t pad_id;
  size_t num_key_value_heads;
  int max_position_embeddings;

  bool tie_word_embeddings = true;

  // The activation function used.
  std::string activation_function{"gelu"};

  // Determines if the model is a moe model.
  bool is_moe = true;
  MoeConfig moe_config;
};

}  // namespace ksana_llm
