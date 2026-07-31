/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/deepseek_v3/deepseek_v3_config.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

inline void PrepareKimiK3Attributes(const nlohmann::json& config_json, ModelConfig& model_config) {
  PrepareDeepSeekV3Attributes(config_json, model_config);
  model_config.moe_config.first_k_dense_replace = config_json.value("first_k_dense_replace", 1);
  model_config.activation_function = config_json.value("hidden_act", "situ");

  // For mla config
  model_config.mla_config.mla_use_nope = config_json.value("mla_use_nope", true);
  model_config.mla_config.mla_use_output_gate = config_json.value("mla_use_output_gate", true);

  // For attn_res config
  model_config.attn_res_config.attn_res_block_size = config_json.value("attn_res_block_size", 0u);
}

struct KimiK3Config : public DeepSeekV3Config {
  virtual ~KimiK3Config() {}

  AttnResConfig attn_res_config;
};

}  // namespace ksana_llm
