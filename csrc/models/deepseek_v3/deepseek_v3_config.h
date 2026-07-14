/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <regex>

#include "models/base/base_model_config.h"
#include "models/common/common_config.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

inline void PrepareDeepSeekV3Attributes(const nlohmann::json& config_json, ModelConfig& model_config) {
  PrepareCommonModelAttributes(config_json, model_config);
  // For moe config
  model_config.moe_config.num_experts = config_json.value("n_routed_experts", 256);
  if (model_config.moe_config.num_experts > 1) {
    model_config.moe_config.use_vllm_moe = true;
    model_config.is_moe = true;
    model_config.moe_config.moe_inter_size = config_json.value("moe_intermediate_size", model_config.inter_size);
    model_config.moe_config.experts_topk = config_json.value("num_experts_per_tok", 8);
    model_config.moe_config.first_k_dense_replace = config_json.value("first_k_dense_replace", 3);
    // For moe group topk config
    model_config.moe_config.num_expert_group = config_json.value("n_group", 1);
    model_config.moe_config.expert_groups_topk = config_json.value("topk_group", 1);
    model_config.moe_config.scoring_func = config_json.value("scoring_func", "sigmoid");
    model_config.moe_config.topk_method = config_json.value("topk_method", "greedy");
    model_config.moe_config.norm_topk_prob = config_json.value("norm_topk_prob", true);
    model_config.moe_config.routed_scaling_factor = config_json.value("routed_scaling_factor", 1.0f);
    if (model_config.moe_config.topk_method == "noaux_tc") {
      model_config.moe_config.use_e_score_correction_bias = true;
    } else {
      model_config.moe_config.use_e_score_correction_bias = false;
    }
  }
  model_config.moe_config.num_shared_experts = config_json.value("n_shared_experts", 1);
  if (model_config.moe_config.num_shared_experts > 0) {
    model_config.has_shared_experts = true;
    model_config.moe_config.shared_expert_inter_size =
        model_config.moe_config.num_shared_experts * model_config.moe_config.moe_inter_size;
  }
  // For mla config
  model_config.use_mla = true;
  if (config_json.contains("q_lora_rank") && config_json["q_lora_rank"].is_number()) {
    model_config.mla_config.q_lora_rank = config_json.value("q_lora_rank", 1536);
  } else {
    model_config.mla_config.q_lora_rank = 0;
  }
  model_config.mla_config.kv_lora_rank = config_json.value("kv_lora_rank", 512);
  model_config.mla_config.qk_nope_head_dim = config_json.value("qk_nope_head_dim", 128);
  model_config.mla_config.qk_rope_head_dim = config_json.value("qk_rope_head_dim", 64);
  model_config.mla_config.v_head_dim = config_json.value("v_head_dim", 128);
  model_config.size_per_head = model_config.mla_config.qk_nope_head_dim + model_config.mla_config.qk_rope_head_dim;
  // For deepseek sparse mla config
  if (model_config.type == "deepseek_v32" || model_config.type == "glm_moe_dsa") {
    model_config.use_dsa = true;
    model_config.dsa_config.index_head_dim = config_json.value("index_head_dim", 128);
    model_config.dsa_config.index_n_heads = config_json.value("index_n_heads", 64);
    model_config.dsa_config.index_topk = config_json.value("index_topk", 2048);
  }
  if (model_config.type == "glm_moe_dsa") {
    model_config.dsa_config.indexer_rope_interleave = config_json.value("indexer_rope_interleave", true);
    model_config.rope_theta =
        config_json.value("rope_parameters", nlohmann::json::object()).value("rope_theta", 1000000);
    model_config.rope_scaling_factor_config.type =
        config_json.value("rope_parameters", nlohmann::json::object()).value("rope_type", "default");
  }
}

struct DeepSeekV3Config : public BaseModelConfig {
  virtual ~DeepSeekV3Config() {}

  bool ContainGptqWeights() const;

  QuantConfig GetGptqQuantConfig();

  bool IsWeightMatchGptq(const std::string& weight_name);

  bool IsGptqContain(const std::string& weight_name);

  std::string type;

  // Type of weight
  DataType weight_data_type;

  std::string tokenizer_path;

  // The max number of (input + output tokens)
  size_t max_token_num;

  size_t max_scheduler_token_num;

  int tensor_para_size;
  int attn_data_para_size;

  // The expert parallel size
  int expert_para_size;
  int moe_tensor_para_size;

  size_t head_num;
  uint32_t size_per_head;
  uint32_t inter_size;
  uint32_t hidden_units;
  uint32_t num_layer;
  uint32_t num_nextn_predict_layers = 0;
  uint32_t rotary_embedding;
  float rope_theta;
  float layernorm_eps;
  uint32_t vocab_size;
  uint32_t start_id;
  std::vector<uint32_t> end_ids;
  uint32_t pad_id;
  size_t num_key_value_heads;
  int max_batch_size;
  int max_position_embeddings;
  size_t block_token_num;
  std::vector<float> k_scales;
  std::vector<float> v_scales;

  RoPEScalingFactor rope_scaling_factor_config;

  bool tie_word_embeddings;
  bool exist_tie_embeddings_param = true;

  // The activation function used.
  std::string activation_function{"swiglu"};

  // Determines if the model is a visual llm model.
  bool is_visual = false;

  // Determines if the model is a quant model.
  bool is_quant;
  QuantConfig quant_config;
  std::vector<QuantConfig> sub_quant_configs;

  // Config for moe
  bool is_moe = false;
  bool has_shared_experts = false;
  MoeConfig moe_config;
  ExpertParallelConfig expert_parallel_config;

  // Config for mla
  bool use_mla = false;
  MlaConfig mla_config;
  // Config for dsa
  bool use_dsa = false;
  DsaConfig dsa_config;

  // others attributes
  std::unordered_map<std::string, std::string> model_attributes;

  ModelFileFormat model_file_format;

  bool load_bias = true;     // Check if load all weights bias.
  int cla_share_factor = 0;  // Determines the number of layers that share k and v.
  bool use_qk_norm = false;  // Check if normlize the attention out q and k.

  std::vector<std::pair<std::regex, std::string>> w4a8_patterns_ = {
      std::make_pair(std::regex(R"(model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.([^.]+)\.weight_scale_inv)"),
                     std::string("model.layers.$1.mlp.experts.$2.$3.scales")),
      std::make_pair(std::regex(R"(model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.([^.]+)\.weight)"),
                     std::string("model.layers.$1.mlp.experts.$2.$3.qweight")),
  };
};

}  // namespace ksana_llm
