/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <vector>

#include "models/base/base_model_config.h"
#include "models/common/common_config.h"
#include "models/deepseek_v3/deepseek_v3_config.h"
#include "nlohmann/json.hpp"
#include "utils/logger.h"

namespace ksana_llm {

// Parse Kimi-K3 LLM backbone attributes (from "text_config", model_type "kimi_linear").
// NOTE: `config_json` here must already be the text_config sub-object.
inline void PrepareKimiK3Attributes(const nlohmann::json& config_json, ModelConfig& model_config) {
  // 1. Common scalar attributes (hidden_size / num_hidden_layers / vocab / eps / tokens / ...).
  PrepareCommonModelAttributes(config_json, model_config);

  // 2. Activation: Kimi-K3 uses the "situ" activation with two beta params.
  model_config.activation_function = config_json.value("hidden_act", "situ");
  model_config.situ_beta = config_json.value("activation_situ_beta", 4.0f);
  model_config.situ_linear_beta = config_json.value("activation_situ_linear_beta", 25.0f);

  // 3. MLA config.
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
  model_config.mla_config.mla_use_nope = config_json.value("mla_use_nope", true);
  model_config.mla_config.mla_use_output_gate = config_json.value("mla_use_output_gate", true);
  model_config.size_per_head = model_config.mla_config.qk_nope_head_dim + model_config.mla_config.qk_rope_head_dim;
  model_config.rotary_embedding = model_config.mla_config.qk_rope_head_dim;

  // 4. MoE config. Kimi-K3 uses different key names than DeepSeek.
  model_config.moe_config.num_experts = config_json.value("num_experts", 896);
  if (model_config.moe_config.num_experts > 1) {
    model_config.moe_config.use_vllm_moe = true;
    model_config.is_moe = true;
    model_config.moe_config.moe_inter_size = config_json.value("moe_intermediate_size", 3072);
    model_config.moe_config.experts_topk = config_json.value("num_experts_per_token", 16);
    model_config.moe_config.first_k_dense_replace = config_json.value("first_k_dense_replace", 1);
    model_config.moe_config.num_expert_group = config_json.value("num_expert_group", 1);
    model_config.moe_config.expert_groups_topk = config_json.value("topk_group", 1);
    model_config.moe_config.scoring_func = config_json.value("moe_router_activation_func", "sigmoid");
    model_config.moe_config.topk_method = config_json.value("topk_method", "noaux_tc");
    model_config.moe_config.norm_topk_prob = config_json.value("moe_renormalize", true);
    model_config.moe_config.routed_scaling_factor = config_json.value("routed_scaling_factor", 1.0f);
    model_config.moe_config.use_e_score_correction_bias = (model_config.moe_config.topk_method == "noaux_tc");
  }
  model_config.moe_config.num_shared_experts = config_json.value("num_shared_experts", 2);
  if (model_config.moe_config.num_shared_experts > 0) {
    model_config.has_shared_experts = true;
    model_config.moe_config.shared_expert_inter_size =
        model_config.moe_config.num_shared_experts * model_config.moe_config.moe_inter_size;
  }
  model_config.moe_config.routed_expert_hidden_size = config_json.value("routed_expert_hidden_size", 3584);
  model_config.moe_config.latent_moe_use_norm = config_json.value("latent_moe_use_norm", true);

  // 5. Kimi Linear (KDA) hybrid attention config.
  if (config_json.contains("linear_attn_config")) {
    const auto& linear_attn_json = config_json["linear_attn_config"];
    auto& kimi_linear_config = model_config.kimi_linear_config;
    kimi_linear_config.head_dim = linear_attn_json.value("head_dim", model_config.mla_config.v_head_dim);
    kimi_linear_config.short_conv_kernel_size = linear_attn_json.value("short_conv_kernel_size", 4u);
    kimi_linear_config.use_full_rank_gate = linear_attn_json.value("use_full_rank_gate", true);
    if (linear_attn_json.contains("gate_lower_bound") && !linear_attn_json["gate_lower_bound"].is_null()) {
      kimi_linear_config.gate_lower_bound = linear_attn_json["gate_lower_bound"].get<float>();
    }

    // Layers using KDA; the remaining ones use full (MLA) attention.
    if (linear_attn_json.contains("kda_layers") && linear_attn_json["kda_layers"].is_array()) {
      for (const auto& layer_idx : linear_attn_json["kda_layers"]) {
        kimi_linear_config.kda_layers.insert(layer_idx.get<uint32_t>());
      }
      if (!kimi_linear_config.kda_layers.empty()) {
        model_config.use_linear_attn = true;
      }
    }

    KLLM_CHECK_WITH_INFO(kimi_linear_config.head_dim == model_config.mla_config.v_head_dim,
                         "linear_attn_config.head_dim must equal v_head_dim");
    KLLM_CHECK_WITH_INFO(linear_attn_json.value("num_heads", model_config.head_num) == model_config.head_num,
                         "linear_attn_config.num_heads must equal num_attention_heads");
  }
  // For attn_res config
  model_config.attn_res_config.attn_res_block_size = config_json.value("attn_res_block_size", 0u);
}

// Executor-side config struct for Kimi-K3. Reuses DeepSeekV3Config for the MLA/MoE
// machinery and adds Kimi Linear (KDA), situ activation and vision fields.
struct KimiK3Config : public DeepSeekV3Config {
  virtual ~KimiK3Config() {}

  // Kimi Linear (KDA) hybrid attention.
  KimiLinearAttnConfig kimi_linear_config;

  // "situ" activation parameters.
  float activation_situ_beta = 0.0f;
  float activation_situ_linear_beta = 0.0f;

  AttnResConfig attn_res_config;
};

}  // namespace ksana_llm
