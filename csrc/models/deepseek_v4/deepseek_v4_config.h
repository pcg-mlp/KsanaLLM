/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <fstream>
#include <regex>
#include <set>
#include <string>

#include "models/base/base_model_config.h"
#include "models/deepseek_v3/deepseek_v3_config.h"
#include "nlohmann/json.hpp"
#include "utils/logger.h"

namespace ksana_llm {

// Infer DSpark stage count from local checkpoint metadata.
//
// The public DSpark top-level config currently keeps num_nextn_predict_layers=1
// even though the attached DSpark module has three mtp.N stages. The HF inference
// directory carries n_mtp_layers; if unavailable, fall back to the safetensors index.
inline int ReadDsparkNumLayers(const std::string& model_path, int default_value) {
  // 1. Try inference/config.json -> "n_mtp_layers"
  const std::string inference_config = model_path + "/inference/config.json";
  {
    std::ifstream file(inference_config);
    if (file.is_open()) {
      try {
        nlohmann::json json_data;
        file >> json_data;
        if (json_data.contains("n_mtp_layers") && !json_data["n_mtp_layers"].is_null()) {
          int n_layers = json_data["n_mtp_layers"].get<int>();
          if (n_layers > 0) {
            return n_layers;
          }
        }
      } catch (const std::exception& e) {
        KLLM_LOG_WARNING << fmt::format("Failed to parse {}: {}", inference_config, e.what());
      }
    }
  }

  // 2. Fall back to model.safetensors.index.json -> count distinct mtp.<N> stages
  const std::string index_path = model_path + "/model.safetensors.index.json";
  {
    std::ifstream file(index_path);
    if (file.is_open()) {
      try {
        nlohmann::json json_data;
        file >> json_data;
        if (json_data.contains("weight_map")) {
          std::set<int> stage_ids;
          for (auto it = json_data["weight_map"].begin(); it != json_data["weight_map"].end(); ++it) {
            const std::string& name = it.key();
            // Split by '.' and check pattern "mtp.<N>.xxx"
            size_t first_dot = name.find('.');
            if (first_dot == std::string::npos || name.substr(0, first_dot) != "mtp") {
              continue;
            }
            size_t second_dot = name.find('.', first_dot + 1);
            if (second_dot == std::string::npos) {
              continue;
            }
            const std::string stage_str = name.substr(first_dot + 1, second_dot - first_dot - 1);
            try {
              stage_ids.insert(std::stoi(stage_str));
            } catch (const std::exception&) {
              // Ignore non-integer stage ids.
            }
          }
          if (!stage_ids.empty()) {
            return *stage_ids.rbegin() + 1;
          }
        }
      } catch (const std::exception& e) {
        KLLM_LOG_WARNING << fmt::format("Failed to parse {}: {}", index_path, e.what());
      }
    }
  }

  return default_value;
}

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

  // For dspark config
  model_config.dspark_config.dspark_block_size = config_json.value("dspark_block_size", 0u);
  model_config.dspark_config.dspark_noise_token_id = config_json.value("dspark_noise_token_id", 0u);
  if (config_json.contains("dspark_target_layer_ids")) {
    auto vec = config_json["dspark_target_layer_ids"].get<std::vector<uint32_t>>();
    model_config.dspark_config.dspark_target_layer_ids =
        std::unordered_set<uint32_t>(vec.begin(), vec.end());
  }
  model_config.dspark_config.dspark_markov_rank = config_json.value("dspark_markov_rank", 256u);

  // Infer DSpark stage count from local checkpoint metadata.
  model_config.num_nextn_predict_layers =
      ReadDsparkNumLayers(model_config.path, model_config.num_nextn_predict_layers);
  KLLM_LOG_INFO << fmt::format("Inferred dsv4 num_nextn_predict_layers: {}", model_config.num_nextn_predict_layers);
}

struct DeepSeekV4Config : public DeepSeekV3Config {
  virtual ~DeepSeekV4Config() {}
  uint32_t num_hash_layers;
  float swiglu_limit = 0.0f;

  bool use_dsv4_attn = false;
  Dsv4AttnConfig dsv4_attn_config;

  bool use_mhc = false;
  MhcConfig mhc_config;

  DsparkConfig dspark_config;
};

}  // namespace ksana_llm
