/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <vector>

#include "configure/environment.h"
#include "device/device_types.h"
#include "models/base/base_model_config.h"

namespace ksana_llm {

// Model config for ChatGLM architecture.
struct ChatglmModelConfig : public BaseModelConfig {
  virtual ~ChatglmModelConfig() {}

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
  std::vector<uint32_t> end_ids;  // ChatGLM may have multiple end tokens
  uint32_t pad_id;
  size_t num_key_value_heads;
  int max_batch_size;
  int max_position_embeddings;
  size_t block_token_num;

  RoPEScalingFactor rope_scaling_factor_config;

  bool tie_word_embeddings;
  bool exist_tie_embeddings_param = true;

  // The activation function used.
  std::string activation_function{"swiglu"};

  // Determines if the model is a visual llm model.
  bool is_visual = false;

  // Determines if the model is a quant model.
  bool is_quant = false;
  QuantConfig quant_config;

  // others attributes
  std::unordered_map<std::string, std::string> model_attributes;
};

}  // namespace ksana_llm
