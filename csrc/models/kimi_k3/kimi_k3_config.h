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

// Parse Kimi-K3 vision tower config (from top-level "vision_config").
inline void PrepareKimiK3VisionAttributes(const nlohmann::json& vision_json, ModelConfig& model_config) {
  auto& v = model_config.kimi_k3_vision_config;
  v.patch_size = vision_json.value("patch_size", 14);
  v.mm_hidden_size = vision_json.value("mm_hidden_size", 1024);
  v.vt_hidden_size = vision_json.value("vt_hidden_size", 1024);
  v.vt_intermediate_size = vision_json.value("vt_intermediate_size", 4096);
  v.vt_num_attention_heads = vision_json.value("vt_num_attention_heads", 12);
  v.vt_num_hidden_layers = vision_json.value("vt_num_hidden_layers", 27);
  v.text_hidden_size = vision_json.value("text_hidden_size", 7168);
  v.qkv_hidden_size = vision_json.value("qkv_hidden_size", 1536);
  v.mm_projector_type = vision_json.value("mm_projector_type", "patchmergerv2");
  v.merge_type = vision_json.value("merge_type", "sd2_tpool");
  if (vision_json.contains("merge_kernel_size")) {
    v.merge_kernel_size = vision_json["merge_kernel_size"].get<std::vector<uint32_t>>();
  }
  v.activation_func = vision_json.value("activation_func", "gelu_pytorch_tanh");
  v.norm_type = vision_json.value("norm_type", "rmsnorm");
  v.projector_ln_eps = vision_json.value("projector_ln_eps", 1e-5f);
  KLLM_LOG_INFO << fmt::format(
      "Kimi-K3 vision: vt_hidden_size: {}, vt_num_hidden_layers: {}, patch_size: {}, mm_projector_type: {}",
      v.vt_hidden_size, v.vt_num_hidden_layers, v.patch_size, v.mm_projector_type);
}

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
    const auto& la = config_json["linear_attn_config"];
    model_config.use_kimi_linear = true;
    if (la.contains("full_attn_layers")) {
      model_config.kimi_linear_config.full_attn_layers = la["full_attn_layers"].get<std::vector<uint32_t>>();
    }
    if (la.contains("kda_layers")) {
      model_config.kimi_linear_config.kda_layers = la["kda_layers"].get<std::vector<uint32_t>>();
    }
    model_config.kimi_linear_config.num_heads = la.value("num_heads", 96);
    model_config.kimi_linear_config.head_dim = la.value("head_dim", 128);
    model_config.kimi_linear_config.short_conv_kernel_size = la.value("short_conv_kernel_size", 4);
    model_config.kimi_linear_config.gate_lower_bound = la.value("gate_lower_bound", -5.0f);
    model_config.kimi_linear_config.use_full_rank_gate = la.value("use_full_rank_gate", true);
  }
  // For attn_res config
  model_config.attn_res_config.attn_res_block_size = config_json.value("attn_res_block_size", 0u);

  KLLM_LOG_INFO << fmt::format(
      "Using kimi_k3 model, num_experts: {}, num_shared_experts: {}, experts_topk: {}, moe_inter_size: {}, "
      "use_kimi_linear: {}, full_attn_layers: {}, kda_layers: {}, activation: {}",
      model_config.moe_config.num_experts, model_config.moe_config.num_shared_experts,
      model_config.moe_config.experts_topk, model_config.moe_config.moe_inter_size, model_config.use_kimi_linear,
      model_config.kimi_linear_config.full_attn_layers.size(), model_config.kimi_linear_config.kda_layers.size(),
      model_config.activation_function);
}

// Executor-side config struct for Kimi-K3. Reuses DeepSeekV3Config for the MLA/MoE
// machinery and adds Kimi Linear (KDA), situ activation and vision fields.
struct KimiK3Config : public DeepSeekV3Config {
  virtual ~KimiK3Config() {}

  // Kimi Linear (KDA) hybrid attention.
  bool use_kimi_linear = false;
  KimiLinearAttnConfig kimi_linear_config;

  // "situ" activation parameters.
  float activation_situ_beta = 0.0f;
  float activation_situ_linear_beta = 0.0f;

  // Vision tower (multimodal).
  bool has_vision = false;
  KimiK3VisionConfig kimi_k3_vision_config;
  uint32_t media_placeholder_token_id = 0;

  AttnResConfig attn_res_config;
};

}  // namespace ksana_llm
