/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <regex>

#include "models/base/base_model_config.h"
#include "models/deepseek_v3/deepseek_v3_config.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

inline void PrepareDeepSeekV4Attributes(const nlohmann::json& config_json, ModelConfig& model_config) {
  PrepareDeepSeekV3Attributes(config_json, model_config);
  // DSV4 has MoE at every layer (no dense MLP prefix); override V3's default of 3.
  model_config.moe_config.first_k_dense_replace = config_json.value("first_k_dense_replace", 0);
  model_config.activation_function = config_json.value("hidden_act", "silu");
  model_config.num_hash_layers = config_json.value("num_hash_layers", 3);
  model_config.size_per_head = config_json.value("head_dim", 512);

  // For dsa config
  model_config.use_dsa = true;
  model_config.dsa_config.index_head_dim = config_json.value("index_head_dim", 128);
  model_config.dsa_config.index_n_heads = config_json.value("index_n_heads", 64);
  model_config.dsa_config.index_topk = config_json.value("index_topk", 2048);
  model_config.dsa_config.indexer_rope_interleave =
      config_json.value("indexer_rope_interleave", true);  // dsv4 uses interleaved rope for indexer

  // For attn config
  model_config.use_dsv4_attn = true;
  model_config.dsv4_attn_config.o_groups = config_json.value("o_groups", 8);
  model_config.dsv4_attn_config.o_lora_rank = config_json.value("o_lora_rank", 1024);
  model_config.dsv4_attn_config.q_lora_rank = config_json.value("q_lora_rank", 1024);
  model_config.dsv4_attn_config.qk_rope_head_dim = config_json.value("qk_rope_head_dim", 64);
  model_config.dsv4_attn_config.sliding_window = config_json.value("sliding_window", 128);
  model_config.dsv4_attn_config.compress_rope_theta = config_json.value("compress_rope_theta", 160000.0f);
  model_config.dsv4_attn_config.compress_ratios = config_json["compress_ratios"].get<std::vector<uint32_t>>();

  // NOTICE(winminkong): dsv4 rotary_embedding 如果用到，可能需要重新求取
  model_config.rotary_embedding = model_config.dsv4_attn_config.qk_rope_head_dim;

  // For mhc config
  model_config.use_mhc = true;
  model_config.mhc_config.hc_mult = config_json.value("hc_mult", 4);
  model_config.mhc_config.hc_sinkhorn_iters = config_json.value("hc_sinkhorn_iters", 20);
  model_config.mhc_config.hc_eps = config_json.value("hc_eps", 1e-6);

  // Add to moe config
  model_config.moe_config.shared_expert_inter_size = model_config.moe_config.moe_inter_size;

  model_config.swiglu_limit = config_json.value("swiglu_limit", 10.0f);
}

struct DeepSeekV4Config : public DeepSeekV3Config {
  virtual ~DeepSeekV4Config() {}
  uint32_t num_hash_layers;
  float swiglu_limit = 0.0f;

  bool use_dsv4_attn = false;
  Dsv4AttnConfig dsv4_attn_config;

  bool use_mhc = false;
  MhcConfig mhc_config;
};

}  // namespace ksana_llm
