/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/environment.h"
#include "models/common/common_config.h"
#include "nlohmann/json.hpp"
#include "utils/logger.h"

namespace ksana_llm {

// The moe experts norm mode.
enum class MoeScaleNormMode { NO_NORM = 0, RE_NORM = 1 };

inline void PrepareQwenMoeAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  PrepareCommonModelAttributes(config_json, model_config);
  model_config.moe_config.num_experts = config_json.value("num_experts", 1);
  if (model_config.moe_config.num_experts > 1) {
    model_config.is_moe = true;
    model_config.moe_config.moe_inter_size =
        config_json.value("moe_intermediate_size", model_config.moe_config.moe_inter_size);
    if (config_json.value("shared_expert_intermediate_size", 0) > 0) {
      model_config.has_shared_experts = true;
      model_config.moe_config.shared_expert_inter_size = config_json["shared_expert_intermediate_size"].get<int>();
    }
    model_config.moe_config.experts_topk = config_json.value("num_experts_per_tok", 2);
    model_config.moe_config.norm_topk_prob = config_json.value("norm_topk_prob", false);
    KLLM_LOG_INFO << fmt::format("Using moe model, num_experts: {}, num_shared_experts: {}, experts_topk: {}",
                                 model_config.moe_config.num_experts, model_config.moe_config.num_shared_experts,
                                 model_config.moe_config.experts_topk);
  }
  model_config.moe_config.apply_weight = false;
}

inline void PrepareMixtralAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  PrepareCommonModelAttributes(config_json, model_config);
  model_config.moe_config.num_experts = config_json.value("num_local_experts", 1);
  if (model_config.moe_config.num_experts > 1) {
    model_config.is_moe = true;
    model_config.has_shared_experts = false;
    model_config.moe_config.moe_inter_size = model_config.inter_size;  //  for mixtral model
    model_config.moe_config.experts_topk = config_json.value("num_experts_per_tok", 2);
    KLLM_LOG_INFO << fmt::format("Using moe model, num_experts: {}, experts_topk: {}",
                                 model_config.moe_config.num_experts, model_config.moe_config.experts_topk);
  }

  model_config.moe_config.apply_weight = false;
}

inline void PrepareHunyuanLargeAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  PrepareCommonModelAttributes(config_json, model_config);
  model_config.moe_config.num_experts = config_json.value("num_experts", 1);
  if (model_config.moe_config.num_experts > 1) {
    model_config.is_moe = true;
    model_config.moe_config.moe_inter_size = model_config.inter_size;
    model_config.moe_config.experts_topk = config_json.value("moe_topk", 1);
  }
  model_config.moe_config.num_shared_experts = config_json.value("num_shared_expert", 0);
  if (model_config.moe_config.num_shared_experts > 0) {
    model_config.has_shared_experts = true;
    model_config.moe_config.shared_expert_inter_size = model_config.inter_size;
  }
  model_config.cla_share_factor = config_json.value("cla_share_factor", 0);
  model_config.use_cla = config_json.value("use_cla", false);
  if (model_config.is_moe && model_config.cla_share_factor != 0)
    model_config.load_bias =
        false;  // The current Hunyuan-large model has saved the bias weights, but does not utilize any bias weights
                // during inference. We will continue to monitor this aspect in the future.
  model_config.use_qk_norm = config_json.value("use_qk_norm", true);
  KLLM_LOG_INFO << fmt::format(
      "Using moe model, num_experts: {}, num_shared_experts: {}, experts_topk: {}, cla_share_factor: {}, use_qk_norm: "
      "{}, load_bias: {}",
      model_config.moe_config.num_experts, model_config.moe_config.num_shared_experts,
      model_config.moe_config.experts_topk, model_config.cla_share_factor, model_config.use_qk_norm,
      model_config.load_bias);

  model_config.moe_config.apply_weight = false;
}

template <typename T>
inline T ExtractPossiblyRepeatedArray(const nlohmann::json &config_json, const std::string &key, T default_value) {
  // This function extracts a value from a JSON object, which may be an array or a single value.
  if (!config_json.contains(key)) {
    return default_value;
  }
  if (config_json[key].is_array()) {
    auto values = config_json[key].get<std::vector<T>>();
    if (values.empty()) {
      return default_value;
    }
    bool all_same = std::adjacent_find(values.begin(), values.end(), std::not_equal_to<>()) == values.end();
    if (!all_same) {
      // TODO(ethanyczeng): if values are not same, we need to do other operation.
      KLLM_LOG_WARNING << "Found different values in " << key << ", using first value: " << values[0];
    }
    return values[0];
  }
  return config_json.value(key, default_value);
}

inline void PrepareHunyuanTurboAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  // TODO(ethanyczeng): 还需要根据HunYuanTurbo中的模型结构进行进一步的适配
  PrepareCommonModelAttributes(config_json, model_config);
  model_config.moe_config.num_experts = config_json.value("num_experts", 1);
  // Handle possibly array-based values with single value check
  model_config.moe_config.experts_topk = ExtractPossiblyRepeatedArray(config_json, "moe_topk", 0);
  model_config.moe_config.num_shared_experts = ExtractPossiblyRepeatedArray(config_json, "num_shared_expert", 0);

  // For arrays that might contain different values but you still need just one representative value
  int moe_inter_size = ExtractPossiblyRepeatedArray(config_json, "moe_intermediate_size", model_config.inter_size);
  model_config.moe_config.moe_inter_size = moe_inter_size;

  if (model_config.moe_config.num_experts > 1 && config_json["moe_topk"].is_number()) {
    model_config.is_moe = true;
    model_config.moe_config.moe_inter_size = model_config.inter_size;
    model_config.moe_config.experts_topk = config_json.value("moe_topk", 1);
  } else if (model_config.moe_config.num_experts == 1) {
    model_config.is_moe = false;
  }
  // model_config.moe_config.num_shared_experts = config_json.value("num_shared_expert", 0);
  KLLM_LOG_DEBUG << fmt::format("num_shared_expert: {}", model_config.moe_config.num_shared_experts);
  if (model_config.moe_config.num_shared_experts > 0 && config_json["num_shared_expert"].is_number()) {
    model_config.has_shared_experts = true;
    model_config.moe_config.shared_expert_inter_size = model_config.inter_size;
  }
  model_config.cla_share_factor = config_json.value("cla_share_factor", 0);
  if (model_config.is_moe && model_config.cla_share_factor != 0)
    model_config.load_bias =
        false;  // The current Hunyuan-large model has saved the bias weights, but does not utilize any bias weights
                // during inference. We will continue to monitor this aspect in the future.
  model_config.use_qk_norm = config_json.value("use_qk_norm", true);
  model_config.mlp_bias = config_json.value("mlp_bias", false);
  model_config.use_cla = config_json.value("use_cla", false);
  KLLM_LOG_INFO << fmt::format("using moe model, num_experts: {}", model_config.moe_config.num_experts);
}

inline void PrepareLlama4Attributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  PrepareCommonModelAttributes(config_json, model_config);
  model_config.moe_config.num_experts = config_json.value("num_local_experts", 1);
  if (model_config.moe_config.num_experts > 1) {
    model_config.is_moe = true;
    model_config.has_shared_experts = true;
    // if layer has moe, layer uses model_config.inter_size as moe_inter_size and shared_expert_inter_size,
    model_config.moe_config.moe_inter_size = model_config.inter_size;
    model_config.moe_config.shared_expert_inter_size = model_config.inter_size;
    // else FFN use intermediate_size_mlp as inter_size for common mlp.
    model_config.inter_size = config_json.value("intermediate_size_mlp", model_config.inter_size);
    model_config.moe_config.experts_topk = config_json.value("num_experts_per_tok", 1);
    KLLM_LOG_INFO << fmt::format("Using moe model, num_experts: {}, experts_topk: {}",
                                 model_config.moe_config.num_experts, model_config.moe_config.experts_topk);
  }
  size_t interleave_moe_layer_step = config_json.value("interleave_moe_layer_step", 1);
  if (config_json.contains("moe_layers") && config_json["moe_layers"].is_array()) {
    model_config.moe_config.moe_layers = config_json["moe_layers"].get<std::vector<size_t>>();
  }
  if (model_config.moe_config.moe_layers.empty()) {
    for (size_t layer_idx = interleave_moe_layer_step - 1; layer_idx < static_cast<size_t>(model_config.num_layer);
         layer_idx += interleave_moe_layer_step) {
      model_config.moe_config.moe_layers.push_back(layer_idx);
    }
  }

  model_config.use_qk_norm = config_json.value("use_qk_norm", true);
  if (config_json.contains("no_rope_layers") && config_json["no_rope_layers"].is_array()) {
    model_config.no_rope_layers = config_json["no_rope_layers"].get<std::vector<size_t>>();
  }
  if (model_config.no_rope_layers.empty()) {
    size_t no_rope_layer_interval = 4;
    for (size_t layer_idx = 0; layer_idx < static_cast<size_t>(model_config.num_layer); layer_idx++) {
      model_config.no_rope_layers.push_back((layer_idx + 1) % no_rope_layer_interval != 0);
    }
  }
  model_config.attn_temperature_tuning = config_json.value("attn_temperature_tuning", 4);
  model_config.attn_scale = config_json.value("attn_scale", 0.1);
  model_config.moe_config.output_router_logits = config_json.value("output_router_logits", false);

  model_config.moe_config.router_aux_loss_coef = config_json.value("router_aux_loss_coef", 0.001);
  model_config.moe_config.router_jitter_noise = config_json.value("router_jitter_noise", 0.0);
  model_config.floor_scale = config_json.value("floor_scale", 8192);

  model_config.moe_config.topk_method = "fast";
  model_config.moe_config.scoring_func = "sigmoid";
  model_config.moe_config.apply_weight = true;
}

inline void PrepareGlm4MoeAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  // First, prepare common model attributes
  PrepareCommonModelAttributes(config_json, model_config);

  // GLM4-MoE uses partial rotary embedding: only a fraction of head_dim gets RoPE
  // PrepareCommonModelAttributes sets rotary_embedding = size_per_head, but we need
  // rotary_embedding = size_per_head * partial_rotary_factor (default 0.5)
  float partial_rotary_factor = config_json.value("partial_rotary_factor", 0.5f);
  model_config.rotary_embedding = static_cast<uint32_t>(model_config.size_per_head * partial_rotary_factor);

  // GLM4-MoE is always a MoE model
  model_config.is_moe = true;

  // Parse MoE config from GLM4-MoE config.json
  model_config.moe_config.num_experts = config_json.value("n_routed_experts", 160);
  model_config.moe_config.experts_topk = config_json.value("num_experts_per_tok", 8);
  model_config.moe_config.moe_inter_size = config_json.value("moe_intermediate_size", model_config.inter_size);
  model_config.moe_config.first_k_dense_replace = config_json.value("first_k_dense_replace", 3);

  // For moe group topk config
  model_config.moe_config.num_expert_group = config_json.value("n_group", 1);
  model_config.moe_config.expert_groups_topk = config_json.value("topk_group", 1);

  // GLM4-MoE uses Sigmoid activation for router
  model_config.moe_config.scoring_func = "sigmoid";

  // GLM4-MoE uses noaux_tc topk_method (same as DeepSeek V3)
  // When use_vllm_moe=true, the topk is computed by InvokeGroupedTopk which
  // doesn't depend on topk_method string, so we can keep the original value
  model_config.moe_config.topk_method = config_json.value("topk_method", "fast");
  model_config.moe_config.norm_topk_prob = config_json.value("norm_topk_prob", true);
  model_config.moe_config.routed_scaling_factor = config_json.value("routed_scaling_factor", 2.5f);

  // GLM4-MoE always uses e_score_correction_bias
  model_config.moe_config.use_e_score_correction_bias = true;
  model_config.moe_config.use_vllm_moe = true;

  // Shared experts config
  model_config.moe_config.num_shared_experts = config_json.value("n_shared_experts", 1);
  if (model_config.moe_config.num_shared_experts > 0) {
    model_config.has_shared_experts = true;
    // Use moe_inter_size to match the model weights
    model_config.moe_config.shared_expert_inter_size =
        model_config.moe_config.num_shared_experts * model_config.moe_config.moe_inter_size;
  }
  model_config.use_qk_norm = false;                          // GLM4-MoE does not need post-RoPE QK Norm
  model_config.enable_qk_pre_norm_before_rotary_pos = true;  // Use pre-RoPE QK Norm
  // // GLM4-MoE specific features
  // model_config.use_qk_norm = config_json.value("use_qk_norm", true);
  model_config.enable_add_qkv_bias = config_json.value("attention_bias", true);
  model_config.load_bias = true;  // GLM4-MoE has bias on QKV projections

  model_config.moe_config.apply_weight = false;

  KLLM_LOG_INFO << fmt::format(
      "GLM4-MoE config: num_experts={}, experts_topk={}, moe_inter_size={}, first_k_dense_replace={}, "
      "has_shared_experts={}, num_shared_experts={}, norm_topk_prob={}, routed_scaling_factor={}, "
      "use_qk_norm={}, enable_add_qkv_bias={}",
      model_config.moe_config.num_experts, model_config.moe_config.experts_topk, model_config.moe_config.moe_inter_size,
      model_config.moe_config.first_k_dense_replace, model_config.has_shared_experts,
      model_config.moe_config.num_shared_experts, model_config.moe_config.norm_topk_prob,
      model_config.moe_config.routed_scaling_factor, model_config.use_qk_norm, model_config.enable_add_qkv_bias);
}
}  // namespace ksana_llm
