/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "configure/model_config.h"
#include "device/device_types.h"
#include "models/base/base_model_config.h"

namespace ksana_llm {

// Model config for mixtral architecture.
struct MixtralConfig : public BaseModelConfig {
  virtual ~MixtralConfig() {}

  // Type of weight
  DataType weight_data_type;

  std::string tokenizer_path;

  // The max number of (input + output tokens)
  size_t max_token_num;

  size_t max_step_token_num;

  size_t tensor_para_size;

  size_t head_num;
  uint32_t size_per_head;
  uint32_t inter_size;
  uint32_t hidden_units;
  uint32_t num_layer;
  uint32_t rotary_embedding;
  float rope_theta;
  float layernorm_eps;
  uint32_t vocab_size;
  uint32_t start_id;
  uint32_t end_id;
  uint32_t pad_id;
  size_t num_key_value_heads;
  int max_batch_size;
  int max_position_embeddings;
  size_t block_token_num;

  RoPEScalingFactor rope_scaling_factor_config;

  bool tie_word_embeddings;
  bool exist_tie_embeddings_param = false;

  // The activation function used.
  std::string activation_function{"silu"};

  // Determines if the model is a quant model.
  bool is_quant;
  QuantConfig quant_config;

  // Determines if the model is a moe model.
  bool is_moe = false;
  bool has_shared_experts = false;
  MoeConfig moe_config;

  // others attributes
  std::unordered_map<std::string, std::string> model_attributes;

  float initializer_range = 0.02f;  // Currently not used for inference
};

}  // namespace ksana_llm
