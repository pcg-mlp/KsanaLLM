/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "configure/model_config.h"
#include "device/device_types.h"
#include "models/base/base_model_config.h"
#include "models/common/common_config.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

inline void PrepareBgeRerankerMinicpmAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  PrepareCommonModelAttributes(config_json, model_config);
  model_config.exist_tie_embeddings_param = true;
  model_config.emb_scale = config_json.value("scale_emb", 1.0f);
  model_config.scale_depth = config_json.value("scale_depth", 1.0f);
  model_config.start_layer = config_json.value("start_layer", 1);
}

// Model config for bge-reranker-minicpm architecture.
struct BgeRerankerMinicpmConfig : public BaseModelConfig {
  virtual ~BgeRerankerMinicpmConfig() = default;

  DataType weight_data_type;

  size_t head_num;
  uint32_t inter_size;
  uint32_t hidden_units;
  uint32_t num_layer;
  uint32_t rotary_embedding;
  float rope_theta;
  float layernorm_eps;
  uint32_t vocab_size;
  uint32_t start_id;
  uint32_t end_id;
  size_t num_key_value_heads;
  int max_position_embeddings;

  // The word embedding scale factor.
  float emb_scale = 1.f;
  // Scaling the hidden states of residual connections.
  float scale_depth = 1.f;
  // For choosing lm_head.linear_head weight.
  int start_layer = 0;

  // Determines if the model is a quant model.
  bool is_quant = false;
  QuantConfig quant_config;
};

}  // namespace ksana_llm
