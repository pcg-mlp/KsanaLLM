/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "configure/model_config.h"
#include "device/device_types.h"
#include "models/base/base_model_config.h"

namespace ksana_llm {

// Model config for qwen2-moe architecture.
struct HunyuanTurboConfig : public BaseModelConfig {
  virtual ~HunyuanTurboConfig() {}

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
  bool exist_tie_embeddings_param = true;

  // The activation function used.
  std::string activation_function{"silu"};

  // Determines if the model is a quant model.
  bool is_quant;
  QuantConfig quant_config;

  // Determines if the model is a moe model.
  bool is_moe = false;
  bool has_shared_experts = false;
  MoeConfig moe_config;

  // Shared expert configuration
  uint32_t shared_expert_intermediate_size;

  // others attributes
  std::unordered_map<std::string, std::string> model_attributes;

  bool load_bias = true;     // Check if load all weights bias.
  int cla_share_factor = 0;  // Determines the number of layers that share k and v.
  bool use_cla = false;
  bool use_qk_norm = false;  // Check if normlize the attention out q and k.
  bool mlp_bias = false;     // Check if use bias in mlp layer.
};

}  // namespace ksana_llm
