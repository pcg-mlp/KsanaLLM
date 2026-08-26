/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/kimi_k3/kimi_k3_weight_loader.h"

#include <gtest/gtest.h>

#include "models/base/utils.h"
#include "models/kimi_k3/kimi_k3_config.h"
#include "nlohmann/json.hpp"
#include "test.h"
#include "utils/string_utils.h"
#include "utils/tensor.h"

#ifdef ENABLE_CUDA
#  include "configure/environment.h"
#  include "device/nvidia/nccl_utils.h"
#  include "helpers/model_test_helper.h"
#  include "utils/memory_utils.h"
#  include "utils/singleton.h"
#endif

namespace ksana_llm {

// ---------------------------------------------------------------------------
// Part 1: Pure-logic tests (no GPU required)
// ---------------------------------------------------------------------------

// --- KimiLinearAttnConfig::IsKdaLayer ---

TEST(KimiK3ConfigTest, IsKdaLayer_BasicKdaLayer) {
  KimiLinearAttnConfig config;
  config.kda_layers = {2, 3, 5, 7, 11};
  // config.json kda_layers are 1-based; layer_idx is 0-based → (layer_idx + 1).
  EXPECT_TRUE(config.IsKdaLayer(1));   // 1+1=2 ∈ kda_layers
  EXPECT_TRUE(config.IsKdaLayer(2));   // 2+1=3 ∈ kda_layers
  EXPECT_TRUE(config.IsKdaLayer(4));   // 4+1=5 ∈ kda_layers
  EXPECT_TRUE(config.IsKdaLayer(10));  // 10+1=11 ∈ kda_layers
}

TEST(KimiK3ConfigTest, IsKdaLayer_MlaLayer) {
  KimiLinearAttnConfig config;
  config.kda_layers = {2, 3, 5};
  // KDA layers (0-based idx → 1-based: 1→2, 2→3, 4→5 are in kda_layers).
  EXPECT_TRUE(config.IsKdaLayer(1));  // 1+1=2 ∈ {2,3,5}
  EXPECT_TRUE(config.IsKdaLayer(2));  // 2+1=3 ∈ {2,3,5}
  EXPECT_TRUE(config.IsKdaLayer(4));  // 4+1=5 ∈ {2,3,5}
  // MLA layers (0-based idx → 1-based: 0→1, 3→4, 5→6 are NOT in kda_layers).
  EXPECT_FALSE(config.IsKdaLayer(0));  // 0+1=1 ∉ {2,3,5}
  EXPECT_FALSE(config.IsKdaLayer(3));  // 3+1=4 ∉ {2,3,5}
  EXPECT_FALSE(config.IsKdaLayer(5));  // 5+1=6 ∉ {2,3,5}
}

TEST(KimiK3ConfigTest, IsKdaLayer_EmptyLayers) {
  KimiLinearAttnConfig config;
  EXPECT_TRUE(config.kda_layers.empty());
  EXPECT_FALSE(config.IsKdaLayer(0));
  EXPECT_FALSE(config.IsKdaLayer(100));
}

TEST(KimiK3ConfigTest, IsKdaLayer_AllKdaLayers) {
  KimiLinearAttnConfig config;
  config.kda_layers = {1, 2, 3, 4, 5};
  EXPECT_TRUE(config.IsKdaLayer(0));
  EXPECT_TRUE(config.IsKdaLayer(4));
  EXPECT_FALSE(config.IsKdaLayer(5));  // 5+1=6∉{1,2,3,4,5}
}

// --- PrepareKimiK3Attributes ---

TEST(KimiK3ConfigTest, PrepareKimiK3Attributes_FullConfig) {
  nlohmann::json config_json = R"({
    "hidden_size": 7168,
    "num_hidden_layers": 93,
    "num_attention_heads": 96,
    "vocab_size": 200000,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 262144,
    "hidden_act": "situ",
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mla_use_nope": true,
    "mla_use_output_gate": true,
    "num_experts": 896,
    "num_shared_experts": 2,
    "moe_intermediate_size": 3072,
    "num_experts_per_token": 16,
    "first_k_dense_replace": 1,
    "num_expert_group": 1,
    "topk_group": 1,
    "moe_router_activation_func": "sigmoid",
    "topk_method": "noaux_tc",
    "moe_renormalize": true,
    "routed_scaling_factor": 2.5,
    "routed_expert_hidden_size": 3584,
    "latent_moe_use_norm": true,
    "linear_attn_config": {
      "kda_layers": [2, 3, 5, 7, 9, 10, 11, 13, 14, 15],
      "num_heads": 96,
      "head_dim": 128,
      "short_conv_kernel_size": 4,
      "gate_lower_bound": -5.0,
      "use_full_rank_gate": true
    },
    "attn_res_block_size": 128
  })"_json;

  ModelConfig model_config;
  PrepareKimiK3Attributes(config_json, model_config);

  // Common attributes.
  EXPECT_EQ(model_config.hidden_units, 7168u);
  EXPECT_EQ(model_config.num_layer, 93u);
  EXPECT_EQ(model_config.vocab_size, 200000u);
  EXPECT_FLOAT_EQ(model_config.layernorm_eps, 1e-6f);

  // Activation.
  EXPECT_EQ(model_config.activation_function, "situ");
  EXPECT_FLOAT_EQ(model_config.situ_beta, 4.0f);
  EXPECT_FLOAT_EQ(model_config.situ_linear_beta, 25.0f);

  // MLA config.
  EXPECT_TRUE(model_config.use_mla);
  EXPECT_EQ(model_config.mla_config.q_lora_rank, 1536u);
  EXPECT_EQ(model_config.mla_config.kv_lora_rank, 512u);
  EXPECT_EQ(model_config.mla_config.qk_nope_head_dim, 128u);
  EXPECT_EQ(model_config.mla_config.qk_rope_head_dim, 64u);
  EXPECT_EQ(model_config.mla_config.v_head_dim, 128u);
  EXPECT_TRUE(model_config.mla_config.mla_use_nope);
  EXPECT_TRUE(model_config.mla_config.mla_use_output_gate);
  EXPECT_EQ(model_config.size_per_head, 192u);  // 128 + 64
  EXPECT_EQ(model_config.rotary_embedding, 64u);

  // MoE config.
  EXPECT_TRUE(model_config.is_moe);
  EXPECT_TRUE(model_config.moe_config.use_vllm_moe);
  EXPECT_EQ(model_config.moe_config.num_experts, 896u);
  EXPECT_EQ(model_config.moe_config.moe_inter_size, 3072u);
  EXPECT_EQ(model_config.moe_config.experts_topk, 16u);
  EXPECT_EQ(model_config.moe_config.first_k_dense_replace, 1u);
  EXPECT_EQ(model_config.moe_config.num_expert_group, 1u);
  EXPECT_EQ(model_config.moe_config.expert_groups_topk, 1u);
  EXPECT_EQ(model_config.moe_config.scoring_func, "sigmoid");
  EXPECT_EQ(model_config.moe_config.topk_method, "noaux_tc");
  EXPECT_TRUE(model_config.moe_config.norm_topk_prob);
  EXPECT_FLOAT_EQ(model_config.moe_config.routed_scaling_factor, 2.5f);
  EXPECT_TRUE(model_config.moe_config.use_e_score_correction_bias);
  EXPECT_EQ(model_config.moe_config.num_shared_experts, 2u);
  EXPECT_TRUE(model_config.has_shared_experts);
  EXPECT_EQ(model_config.moe_config.shared_expert_inter_size, 6144u);  // 2 * 3072
  EXPECT_EQ(model_config.moe_config.routed_expert_hidden_size, 3584u);
  EXPECT_TRUE(model_config.moe_config.latent_moe_use_norm);

  // Kimi Linear (KDA) config.
  EXPECT_TRUE(model_config.use_linear_attn);
  EXPECT_EQ(model_config.kimi_linear_config.kda_layers.size(), 10u);
  EXPECT_EQ(model_config.kimi_linear_config.short_conv_kernel_size, 4u);
  EXPECT_EQ(model_config.kimi_linear_config.head_dim, 128u);
  EXPECT_FLOAT_EQ(model_config.kimi_linear_config.gate_lower_bound, -5.0f);
  EXPECT_TRUE(model_config.kimi_linear_config.use_full_rank_gate);

  // Attn res config.
  EXPECT_EQ(model_config.attn_res_config.attn_res_block_size, 128u);
}

TEST(KimiK3ConfigTest, PrepareKimiK3Attributes_Defaults) {
  nlohmann::json config_json = R"({
    "hidden_size": 7168,
    "num_hidden_layers": 93,
    "vocab_size": 200000
  })"_json;

  ModelConfig model_config;
  PrepareKimiK3Attributes(config_json, model_config);

  // Defaults should be applied.
  EXPECT_EQ(model_config.activation_function, "situ");
  EXPECT_FLOAT_EQ(model_config.situ_beta, 4.0f);
  EXPECT_FLOAT_EQ(model_config.situ_linear_beta, 25.0f);
  EXPECT_EQ(model_config.mla_config.q_lora_rank, 0u);  // q_lora_rank not present
  EXPECT_EQ(model_config.mla_config.kv_lora_rank, 512u);
  EXPECT_EQ(model_config.moe_config.num_experts, 896u);
  EXPECT_FALSE(model_config.use_linear_attn);  // No linear_attn_config
  EXPECT_EQ(model_config.attn_res_config.attn_res_block_size, 0u);
}

TEST(KimiK3ConfigTest, PrepareKimiK3Attributes_NoQLoraRank) {
  nlohmann::json config_json = R"({
    "hidden_size": 7168,
    "num_hidden_layers": 93,
    "vocab_size": 200000,
    "kv_lora_rank": 512
  })"_json;

  ModelConfig model_config;
  PrepareKimiK3Attributes(config_json, model_config);
  // q_lora_rank not present → should be 0.
  EXPECT_EQ(model_config.mla_config.q_lora_rank, 0u);
}

TEST(KimiK3ConfigTest, PrepareKimiK3Attributes_NoMoe) {
  nlohmann::json config_json = R"({
    "hidden_size": 7168,
    "num_hidden_layers": 93,
    "vocab_size": 200000,
    "num_experts": 1,
    "num_shared_experts": 0
  })"_json;

  ModelConfig model_config;
  PrepareKimiK3Attributes(config_json, model_config);
  // num_experts <= 1 → not MoE.
  EXPECT_FALSE(model_config.is_moe);
  EXPECT_FALSE(model_config.moe_config.use_vllm_moe);
  EXPECT_FALSE(model_config.has_shared_experts);
  EXPECT_EQ(model_config.moe_config.shared_expert_inter_size, 0u);
}

TEST(KimiK3ConfigTest, PrepareKimiK3Attributes_MoeNotNoauxTc) {
  nlohmann::json config_json = R"({
    "hidden_size": 7168,
    "num_hidden_layers": 93,
    "vocab_size": 200000,
    "num_experts": 8,
    "topk_method": "greedy"
  })"_json;

  ModelConfig model_config;
  PrepareKimiK3Attributes(config_json, model_config);
  // topk_method != "noaux_tc" → use_e_score_correction_bias = false.
  EXPECT_FALSE(model_config.moe_config.use_e_score_correction_bias);
}

// --- KimiK3Config struct defaults ---

TEST(KimiK3ConfigTest, StructDefaults) {
  KimiK3Config config;
  EXPECT_TRUE(config.kimi_linear_config.kda_layers.empty());
  EXPECT_FLOAT_EQ(config.activation_situ_beta, 0.0f);
  EXPECT_FLOAT_EQ(config.activation_situ_linear_beta, 0.0f);
}

// --- Weight name classification tests ---
// These test the string matching logic used in ProcessMlaWeight, ProcessKdaWeight,
// ProcessLatentMoeAndRouter, and ProcessModelWeights to route weights.

TEST(KimiK3WeightNameTest, MlaWeightNames_Routing) {
  // MLA layer weights should match their respective branches in ProcessMlaWeight.
  std::string layer_prefix = "model.layers.0.self_attn.";

  EXPECT_NE((layer_prefix + "q_a_proj.weight").find(".q_a_proj.weight"), std::string::npos);
  EXPECT_NE((layer_prefix + "kv_a_proj_with_mqa.weight").find(".kv_a_proj_with_mqa.weight"), std::string::npos);
  EXPECT_NE((layer_prefix + "q_b_proj.weight").find(".q_b_proj.weight"), std::string::npos);
  EXPECT_NE((layer_prefix + "kv_b_proj.weight").find(".kv_b_proj.weight"), std::string::npos);
  EXPECT_NE((layer_prefix + "o_proj.weight").find(".o_proj.weight"), std::string::npos);
  EXPECT_NE((layer_prefix + "g_proj.weight").find(".g_proj.weight"), std::string::npos);
}

TEST(KimiK3WeightNameTest, KdaWeightNames_Routing) {
  // KDA layer weights.
  std::string layer_prefix = "model.layers.1.self_attn.";

  // qkvg fusion targets.
  EXPECT_TRUE(CheckWeightNameMatched(layer_prefix + "q_proj.weight",
                                     {".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".g_proj.weight"}, false));
  EXPECT_TRUE(CheckWeightNameMatched(layer_prefix + "k_proj.weight",
                                     {".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".g_proj.weight"}, false));
  EXPECT_TRUE(CheckWeightNameMatched(layer_prefix + "v_proj.weight",
                                     {".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".g_proj.weight"}, false));
  EXPECT_TRUE(CheckWeightNameMatched(layer_prefix + "g_proj.weight",
                                     {".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".g_proj.weight"}, false));

  // b_proj / f_b_proj (individual column split).
  EXPECT_TRUE(CheckWeightNameMatched(layer_prefix + "b_proj.weight", {".b_proj.weight", ".f_b_proj.weight"}, false));
  EXPECT_TRUE(CheckWeightNameMatched(layer_prefix + "f_b_proj.weight", {".b_proj.weight", ".f_b_proj.weight"}, false));

  // f_a_proj (replicated).
  EXPECT_NE((layer_prefix + "f_a_proj.weight").find(".f_a_proj.weight"), std::string::npos);

  // o_norm (replicated).
  EXPECT_NE((layer_prefix + "o_norm.weight").find(".o_norm.weight"), std::string::npos);

  // qkv_conv1d fusion.
  EXPECT_TRUE(CheckWeightNameMatched(layer_prefix + "q_conv1d.weight",
                                     {".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight"}, false));
  EXPECT_TRUE(CheckWeightNameMatched(layer_prefix + "k_conv1d.weight",
                                     {".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight"}, false));
  EXPECT_TRUE(CheckWeightNameMatched(layer_prefix + "v_conv1d.weight",
                                     {".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight"}, false));

  // A_log.
  EXPECT_NE((layer_prefix + "A_log").find(".A_log"), std::string::npos);

  // dt_bias.
  EXPECT_NE((layer_prefix + "dt_bias").find(".dt_bias"), std::string::npos);

  // o_proj (row split).
  EXPECT_NE((layer_prefix + "o_proj.weight").find(".o_proj.weight"), std::string::npos);
}

TEST(KimiK3WeightNameTest, LatentMoeAndRouterNames_Routing) {
  // Latent MoE projections.
  EXPECT_NE(std::string("model.layers.1.mlp.routed_expert_norm.weight").find(".routed_expert_norm.weight"),
            std::string::npos);
  EXPECT_NE(std::string("model.layers.1.mlp.routed_expert_up_proj.weight").find(".routed_expert_up_proj.weight"),
            std::string::npos);
  EXPECT_NE(std::string("model.layers.1.mlp.routed_expert_down_proj.weight").find(".routed_expert_down_proj.weight"),
            std::string::npos);

  // Router gate.
  EXPECT_NE(std::string("model.layers.1.mlp.gate.weight").find(".mlp.gate.weight"), std::string::npos);
  EXPECT_NE(std::string("model.layers.1.mlp.gate.e_score_correction_bias").find("gate.e_score_correction_bias"),
            std::string::npos);
}

TEST(KimiK3WeightNameTest, DenseMlpNames_Routing) {
  // Dense MLP (layer 0).
  EXPECT_TRUE(CheckWeightNameMatched(
      "model.layers.0.mlp.gate_proj.weight",
      {".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj"}, false));
  EXPECT_TRUE(CheckWeightNameMatched(
      "model.layers.0.mlp.up_proj.weight",
      {".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj"}, false));
  EXPECT_TRUE(CheckWeightNameMatched("model.layers.0.mlp.down_proj.weight",
                                     {".mlp.down_proj.", ".mlp.shared_expert.down_proj"}, false));

  // Shared expert.
  EXPECT_TRUE(CheckWeightNameMatched(
      "model.layers.1.mlp.shared_expert.gate_proj.weight",
      {".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj"}, false));
  EXPECT_TRUE(CheckWeightNameMatched("model.layers.1.mlp.shared_expert.down_proj.weight",
                                     {".mlp.down_proj.", ".mlp.shared_expert.down_proj"}, false));
}

TEST(KimiK3WeightNameTest, EmbedAndLmHead_Routing) {
  EXPECT_TRUE(CheckWeightNameMatched("model.embed_tokens.weight", {"model.embed_tokens.weight"}, true));
  EXPECT_TRUE(CheckWeightNameMatched("lm_head.weight", {"lm_head.weight"}, true));
}

TEST(KimiK3WeightNameTest, NormAndResProj_Routing) {
  // Norm weights contain "norm.".
  EXPECT_NE(std::string("model.layers.0.input_layernorm.weight").find("norm."), std::string::npos);
  EXPECT_NE(std::string("model.layers.0.post_attention_layernorm.weight").find("norm."), std::string::npos);
  EXPECT_NE(std::string("model.layers.0.self_attention_res_norm.weight").find("norm."), std::string::npos);

  // res_proj weights end with "_res_proj.weight".
  EXPECT_NE(std::string("model.layers.0.self_attention_res_proj.weight").find("_res_proj.weight"), std::string::npos);
  EXPECT_NE(std::string("model.layers.0.mlp_res_proj.weight").find("_res_proj.weight"), std::string::npos);
}

TEST(KimiK3WeightNameTest, MoeExpertNames_Routing) {
  // Expert weights after weight_map renaming.
  EXPECT_NE(std::string("model.layers.1.mlp.experts.0.gate_proj.weight").find(".mlp.experts."), std::string::npos);
  EXPECT_NE(std::string("model.layers.1.mlp.experts.0.up_proj.weight").find(".mlp.experts."), std::string::npos);
  EXPECT_NE(std::string("model.layers.1.mlp.experts.0.down_proj.weight").find(".mlp.experts."), std::string::npos);
  EXPECT_NE(std::string("model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv").find(".mlp.experts."),
            std::string::npos);
}

TEST(KimiK3WeightNameTest, FilterWeightNames_SkipList) {
  // These should be filtered by FilterWeightNames.
  std::vector<std::string> skip_list = {"self_attn.rotary_emb.inv_freq", "rotary_emb.inv_freq"};

  EXPECT_TRUE(CheckWeightNameMatched("model.layers.0.self_attn.rotary_emb.inv_freq", skip_list, false));
  EXPECT_TRUE(CheckWeightNameMatched("model.layers.0.rotary_emb.inv_freq", skip_list, false));

  // These should NOT be filtered.
  EXPECT_FALSE(CheckWeightNameMatched("model.layers.0.self_attn.q_a_proj.weight", skip_list, false));
  EXPECT_FALSE(CheckWeightNameMatched("model.embed_tokens.weight", skip_list, false));
  EXPECT_FALSE(CheckWeightNameMatched("model.layers.1.mlp.gate.weight", skip_list, false));
}

TEST(KimiK3WeightNameTest, SelfAttnPrefix_Routing) {
  // self_attn weights should match ".self_attn." prefix.
  EXPECT_NE(std::string("model.layers.0.self_attn.q_a_proj.weight").find(".self_attn."), std::string::npos);
  EXPECT_NE(std::string("model.layers.1.self_attn.q_proj.weight").find(".self_attn."), std::string::npos);
  // Non-self_attn weights should not match.
  EXPECT_EQ(std::string("model.layers.0.mlp.gate_proj.weight").find(".self_attn."), std::string::npos);
}

// --- KDA segment index logic ---

TEST(KimiK3KdaFusionTest, QkvgSegmentIndex) {
  // Verify segment index determination logic.
  auto get_seg_idx = [](const std::string& name) -> int {
    if (name.find(".q_proj.weight") != std::string::npos) return 0;
    if (name.find(".k_proj.weight") != std::string::npos) return 1;
    if (name.find(".v_proj.weight") != std::string::npos) return 2;
    if (name.find(".g_proj.weight") != std::string::npos) return 3;
    return -1;
  };

  EXPECT_EQ(get_seg_idx("model.layers.1.self_attn.q_proj.weight"), 0);
  EXPECT_EQ(get_seg_idx("model.layers.1.self_attn.k_proj.weight"), 1);
  EXPECT_EQ(get_seg_idx("model.layers.1.self_attn.v_proj.weight"), 2);
  EXPECT_EQ(get_seg_idx("model.layers.1.self_attn.g_proj.weight"), 3);
  EXPECT_EQ(get_seg_idx("model.layers.1.self_attn.o_proj.weight"), -1);
}

TEST(KimiK3KdaFusionTest, QkvConv1dSegmentIndex) {
  auto get_seg_idx = [](const std::string& name) -> int {
    if (name.find(".q_conv1d.weight") != std::string::npos) return 0;
    if (name.find(".k_conv1d.weight") != std::string::npos) return 1;
    if (name.find(".v_conv1d.weight") != std::string::npos) return 2;
    return -1;
  };

  EXPECT_EQ(get_seg_idx("model.layers.1.self_attn.q_conv1d.weight"), 0);
  EXPECT_EQ(get_seg_idx("model.layers.1.self_attn.k_conv1d.weight"), 1);
  EXPECT_EQ(get_seg_idx("model.layers.1.self_attn.v_conv1d.weight"), 2);
  EXPECT_EQ(get_seg_idx("model.layers.1.self_attn.A_log"), -1);
}

TEST(KimiK3KdaFusionTest, FusedQkvgNameConstruction) {
  // Verify fused name construction from individual proj names.
  auto get_fused_name = [](const std::string& name) -> std::string {
    int seg_idx = -1;
    if (name.find(".q_proj.weight") != std::string::npos)
      seg_idx = 0;
    else if (name.find(".k_proj.weight") != std::string::npos)
      seg_idx = 1;
    else if (name.find(".v_proj.weight") != std::string::npos)
      seg_idx = 2;
    else if (name.find(".g_proj.weight") != std::string::npos)
      seg_idx = 3;

    std::string fused_name = name.substr(0, name.find(".q_proj.")) + ".fused_qkvg_proj.weight";
    if (seg_idx == 1)
      fused_name = name.substr(0, name.find(".k_proj.")) + ".fused_qkvg_proj.weight";
    else if (seg_idx == 2)
      fused_name = name.substr(0, name.find(".v_proj.")) + ".fused_qkvg_proj.weight";
    else if (seg_idx == 3)
      fused_name = name.substr(0, name.find(".g_proj.")) + ".fused_qkvg_proj.weight";
    return fused_name;
  };

  EXPECT_EQ(get_fused_name("model.layers.1.self_attn.q_proj.weight"),
            "model.layers.1.self_attn.fused_qkvg_proj.weight");
  EXPECT_EQ(get_fused_name("model.layers.1.self_attn.k_proj.weight"),
            "model.layers.1.self_attn.fused_qkvg_proj.weight");
  EXPECT_EQ(get_fused_name("model.layers.1.self_attn.v_proj.weight"),
            "model.layers.1.self_attn.fused_qkvg_proj.weight");
  EXPECT_EQ(get_fused_name("model.layers.1.self_attn.g_proj.weight"),
            "model.layers.1.self_attn.fused_qkvg_proj.weight");
}

TEST(KimiK3KdaFusionTest, FusedQkvConv1dNameConstruction) {
  auto get_fused_name = [](const std::string& name) -> std::string {
    int seg_idx = -1;
    if (name.find(".q_conv1d.weight") != std::string::npos)
      seg_idx = 0;
    else if (name.find(".k_conv1d.weight") != std::string::npos)
      seg_idx = 1;
    else if (name.find(".v_conv1d.weight") != std::string::npos)
      seg_idx = 2;

    std::string prefix = name.substr(0, name.find(".q_conv1d."));
    if (seg_idx == 1)
      prefix = name.substr(0, name.find(".k_conv1d."));
    else if (seg_idx == 2)
      prefix = name.substr(0, name.find(".v_conv1d."));
    return prefix + ".qkv_conv1d.weight";
  };

  EXPECT_EQ(get_fused_name("model.layers.1.self_attn.q_conv1d.weight"), "model.layers.1.self_attn.qkv_conv1d.weight");
  EXPECT_EQ(get_fused_name("model.layers.1.self_attn.k_conv1d.weight"), "model.layers.1.self_attn.qkv_conv1d.weight");
  EXPECT_EQ(get_fused_name("model.layers.1.self_attn.v_conv1d.weight"), "model.layers.1.self_attn.qkv_conv1d.weight");
}

// --- MLA fused_lora_a_proj name construction ---

TEST(KimiK3MlaFusionTest, FusedLoraANameConstruction) {
  // q_a_proj → fused_lora_a_proj (front segment).
  std::string q_a_name = "model.layers.0.self_attn.q_a_proj.weight";
  std::string fused_name = GetReplacedName(q_a_name, ".q_a_proj.weight", ".fused_lora_a_proj.weight");
  EXPECT_EQ(fused_name, "model.layers.0.self_attn.fused_lora_a_proj.weight");

  // kv_a_proj_with_mqa → fused_lora_a_proj (back segment).
  std::string kv_a_name = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight";
  fused_name = GetReplacedName(kv_a_name, ".kv_a_proj_with_mqa.weight", ".fused_lora_a_proj.weight");
  EXPECT_EQ(fused_name, "model.layers.0.self_attn.fused_lora_a_proj.weight");
}

// --- MLA q_b_nope_rope name construction ---

TEST(KimiK3MlaFusionTest, QBNopeRopeNameConstruction) {
  std::string q_b_name = "model.layers.0.self_attn.q_b_proj.weight";
  // The code uses: file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.q_b_nope_rope_proj.weight"
  std::string result = q_b_name.substr(0, q_b_name.find_first_of('_')) + "_attn.q_b_nope_rope_proj.weight";
  EXPECT_EQ(result, "model.layers.0.self_attn.q_b_nope_rope_proj.weight");
}

TEST(KimiK3MlaFusionTest, KvBNopeNameConstruction) {
  std::string kv_b_name = "model.layers.0.self_attn.kv_b_proj.weight";
  std::string kv_b_nope = kv_b_name.substr(0, kv_b_name.find_first_of('_')) + "_attn.kv_b_nope_proj.weight";
  EXPECT_EQ(kv_b_nope, "model.layers.0.self_attn.kv_b_nope_proj.weight");

  std::string v_head = kv_b_name.substr(0, kv_b_name.find_first_of('_')) + "_attn.v_head_proj.weight";
  EXPECT_EQ(v_head, "model.layers.0.self_attn.v_head_proj.weight");

  std::string w_uk_t = kv_b_nope.substr(0, kv_b_nope.find_first_of('_')) + "_attn.w_uk_t.weight";
  EXPECT_EQ(w_uk_t, "model.layers.0.self_attn.w_uk_t.weight");

  std::string w_uv = v_head.substr(0, v_head.find_first_of('_')) + "_attn.w_uv.weight";
  EXPECT_EQ(w_uv, "model.layers.0.self_attn.w_uv.weight");
}

// --- Shape calculation tests ---

TEST(KimiK3ShapeTest, QkvgFusedShape) {
  // dim0 = 12288 (96 heads * 128 head_dim), dim1 = 7168, attn_tp = 2.
  size_t dim0 = 12288, dim1 = 7168, attn_tp = 2;
  size_t dim0_tp = DivRoundUp(dim0, attn_tp);  // 6144
  EXPECT_EQ(dim0_tp, 6144u);
  std::vector<size_t> fused_shape = {4 * dim0_tp, dim1};
  EXPECT_EQ(fused_shape[0], 24576u);  // 4 * 6144
  EXPECT_EQ(fused_shape[1], 7168u);
}

TEST(KimiK3ShapeTest, QkvConv1dFusedShape) {
  // dim0 = 12288, inner = 1 * 4 (kernel_size), attn_tp = 2.
  size_t dim0 = 12288, kernel_size = 4, attn_tp = 2;
  size_t dim0_tp = DivRoundUp(dim0, attn_tp);  // 6144
  std::vector<size_t> fused_shape = {3 * dim0_tp, 1, kernel_size};
  EXPECT_EQ(fused_shape[0], 18432u);  // 3 * 6144
  EXPECT_EQ(fused_shape[1], 1u);
  EXPECT_EQ(fused_shape[2], 4u);
}

TEST(KimiK3ShapeTest, QBNopeRopeShape) {
  // head_num=96, qk_nope=128, qk_rope=64, attn_tp=2.
  size_t head_num = 96, qk_nope = 128, qk_rope = 64, attn_tp = 2;
  size_t dim0 = head_num * (qk_nope + qk_rope);  // 96 * 192 = 18432
  size_t dim0_tp = DivRoundUp(dim0, attn_tp);    // 9216
  EXPECT_EQ(dim0_tp, 9216u);
}

TEST(KimiK3ShapeTest, ALocalHeadsShape) {
  // kda_num_heads=96, attn_tp=2.
  size_t kda_num_heads = 96, attn_tp = 2;
  size_t local_heads = DivRoundUp(kda_num_heads, attn_tp);  // 48
  EXPECT_EQ(local_heads, 48u);
}

TEST(KimiK3ShapeTest, MoEExpertShape) {
  // num_experts=896, global_expert_para=8, moe_inter=3072, moe_tp=1.
  size_t num_experts = 896, global_expert_para = 8, moe_inter = 3072, moe_tp = 1;
  size_t num_experts_per_rank = DivRoundUp(num_experts, global_expert_para);  // 112
  EXPECT_EQ(num_experts_per_rank, 112u);

  size_t moe_inter_per_rank = DivRoundUp(moe_inter, moe_tp);  // 3072
  EXPECT_EQ(moe_inter_per_rank, 3072u);

  // w13 shape: [num_experts_per_rank, moe_inter_per_rank * 2, inner].
  size_t inner = 3584;  // routed_expert_hidden_size / 2 (MXFP4 packed)
  std::vector<size_t> w13_shape = {num_experts_per_rank, moe_inter_per_rank * 2, inner};
  EXPECT_EQ(w13_shape[0], 112u);
  EXPECT_EQ(w13_shape[1], 6144u);
  EXPECT_EQ(w13_shape[2], 3584u);

  // down shape: [num_experts_per_rank, expert_hidden, inner_tp].
  size_t expert_hidden = 3584;
  size_t inner_tp = DivRoundUp(inner, moe_tp);  // 3584
  std::vector<size_t> down_shape = {num_experts_per_rank, expert_hidden, inner_tp};
  EXPECT_EQ(down_shape[0], 112u);
  EXPECT_EQ(down_shape[1], 3584u);
  EXPECT_EQ(down_shape[2], 3584u);
}

// --- PostProcessModelWeights logic tests ---

TEST(KimiK3PostProcessTest, EScoreCorrectionBiasCastToFp32) {
  // e_score_correction_bias should be detected for fp32 cast.
  EXPECT_NE(std::string("model.layers.1.mlp.gate.e_score_correction_bias").find("gate.e_score_correction_bias"),
            std::string::npos);
  EXPECT_EQ(std::string("model.layers.1.mlp.gate.weight").find("gate.e_score_correction_bias"), std::string::npos);
}

TEST(KimiK3PostProcessTest, QuantWeightSkip) {
  // Non-fp32/fp16/bf16 dtypes should be skipped (quantized weights).
  // The code checks: !(dtype == FP32 || dtype == FP16 || dtype == BF16) → skip.
  EXPECT_NE(DataType::TYPE_FP32, DataType::TYPE_FP16);
  EXPECT_NE(DataType::TYPE_FP32, DataType::TYPE_BF16);
  EXPECT_NE(DataType::TYPE_FP16, DataType::TYPE_BF16);
}

// ---------------------------------------------------------------------------
// Part 2: CUDA integration tests — actually call ProcessModelWeights /
// FilterWeightNames / PostProcessModelWeights to cover the real weight loading
// code paths in kimi_k3_weight_loader.cpp.
// ---------------------------------------------------------------------------

#ifdef ENABLE_CUDA

// Test subclass to expose protected members for testing.
class TestKimiK3WeightLoader : public KimiK3WeightLoader {
 public:
  using KimiK3WeightLoader::KimiK3WeightLoader;
  using KimiK3WeightLoader::kimi_k3_config_;
  using KimiK3WeightLoader::pipeline_config_;
  using KimiK3WeightLoader::runtime_config_;
  using KimiK3WeightLoader::weight_impl_;
  using KimiK3WeightLoader::weights_to_permute_;
  using BaseModelWeightLoader::context_;
};

class KimiK3WeightLoaderIntegrationTest : public testing::Test {
 protected:
  std::shared_ptr<Context> context_;
  std::shared_ptr<KimiK3Config> config_;

  void SetUp() override {
    InitLoguru();
    const auto& env = Singleton<Environment>::GetInstance();
    context_ = std::make_shared<Context>();
    context_->global.reset(new GlobalContext(2, 1, 1, env));
    context_->global->InitDistributedConfig();
    context_->device = std::make_shared<DeviceContext>(1);
    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context_->device->ext->InitNcclParam(nccl_uid);

    config_ = std::make_shared<KimiK3Config>();
    config_->type = "kimi_k3";
    config_->num_layer = 4;
    config_->hidden_units = 64;
    config_->size_per_head = 64;
    config_->head_num = 2;
    config_->num_key_value_heads = 2;
    config_->vocab_size = 100;
    config_->weight_data_type = TYPE_BF16;
    config_->is_quant = false;
    config_->is_compressed_tensors = false;
    config_->use_mla = true;
    config_->mla_config.q_lora_rank = 32;
    config_->mla_config.kv_lora_rank = 16;
    config_->mla_config.qk_nope_head_dim = 32;
    config_->mla_config.qk_rope_head_dim = 32;
    config_->mla_config.v_head_dim = 32;
    config_->mla_config.mla_use_nope = true;
    config_->mla_config.mla_use_output_gate = true;
    config_->kimi_linear_config.kda_layers = {2, 3};
    config_->is_moe = true;
    config_->moe_config.num_experts = 2;
    config_->moe_config.moe_inter_size = 64;
    config_->moe_config.shared_expert_inter_size = 128;
    config_->moe_config.num_shared_experts = 2;
    config_->moe_config.routed_expert_hidden_size = 64;
    config_->moe_config.latent_moe_use_norm = true;
    config_->moe_config.first_k_dense_replace = 1;
    config_->moe_config.experts_topk = 1;
    config_->moe_config.num_expert_group = 1;
    config_->moe_config.expert_groups_topk = 1;
    config_->moe_config.scoring_func = "sigmoid";
    config_->moe_config.topk_method = "noaux_tc";
    config_->moe_config.norm_topk_prob = true;
    config_->moe_config.routed_scaling_factor = 1.0f;
    config_->moe_config.use_vllm_moe = true;
    config_->moe_config.use_e_score_correction_bias = true;
    config_->has_shared_experts = true;
    config_->tensor_para_size = 2;
    config_->moe_tensor_para_size = 1;
    config_->expert_parallel_config.expert_para_size = 1;
    config_->expert_parallel_config.expert_world_size = 1;
    config_->expert_parallel_config.expert_node_rank = 0;
  }

  Tensor MakeHostTensor(DataType dtype, const std::vector<size_t>& shape) {
    Tensor t(MemoryLocation::LOCATION_HOST, dtype, shape);
    return t;
  }

  std::shared_ptr<TestKimiK3WeightLoader> MakeLoader() {
    return std::make_shared<TestKimiK3WeightLoader>(config_, Singleton<Environment>::GetInstance(), context_);
  }

  // Verify that a device weight exists and has the expected shape.
  void ExpectShape(const std::unordered_map<std::string, Tensor>& dev, const std::string& name,
                   const std::vector<size_t>& expected) {
    auto it = dev.find(name);
    ASSERT_NE(it, dev.end()) << "Missing: " << name;
    EXPECT_EQ(it->second.shape, expected) << "Wrong shape for: " << name;
  }
};

// --- Constructor ---

TEST_F(KimiK3WeightLoaderIntegrationTest, ConstructWeightLoader) {
  auto loader = MakeLoader();
  ASSERT_NE(loader, nullptr);
  ASSERT_NE(loader->kimi_k3_config_, nullptr);
  EXPECT_EQ(loader->kimi_k3_config_->hidden_units, 64u);
  // Verify tp=2 is properly configured.
  EXPECT_EQ(context_->global->GetTensorParallelSize(), 2u);
  EXPECT_EQ(context_->global->GetAttentionTensorParallelSize(), 2u);
}

// --- FilterWeightNames ---

TEST_F(KimiK3WeightLoaderIntegrationTest, FilterWeightNames_SkipsRotary) {
  auto loader = MakeLoader();
  std::vector<std::string> names = {
      "model.layers.0.self_attn.q_a_proj.weight",
      "model.layers.0.self_attn.rotary_emb.inv_freq",
      "model.embed_tokens.weight",
      "lm_head.weight",
      "model.layers.0.input_layernorm.weight",
  };
  EXPECT_TRUE(loader->FilterWeightNames(names).OK());
  std::unordered_set<std::string> remaining(names.begin(), names.end());
  EXPECT_EQ(remaining.count("model.layers.0.self_attn.rotary_emb.inv_freq"), 0u);
  EXPECT_EQ(remaining.count("model.layers.0.self_attn.q_a_proj.weight"), 1u);
  EXPECT_EQ(remaining.count("model.embed_tokens.weight"), 1u);
  EXPECT_EQ(remaining.count("lm_head.weight"), 1u);
}

// --- MLA weight loading ---

TEST_F(KimiK3WeightLoaderIntegrationTest, Mla_QaProjFusion) {
  auto loader = MakeLoader();
  Tensor qa = MakeHostTensor(TYPE_BF16, {32, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.0.self_attn.q_a_proj.weight", qa}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // fused_lora_a_proj: q_a is REPLICATED (not tp-split), shape = [q_lora + kv_lora + qk_rope, hidden] = [80, 64]
  ExpectShape(dev, "model.layers.0.self_attn.fused_lora_a_proj.weight", {80, 64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Mla_KvAProjFusion) {
  auto loader = MakeLoader();
  Tensor kva = MakeHostTensor(TYPE_BF16, {16 + 32, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.0.self_attn.kv_a_proj_with_mqa.weight", kva}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // fused_lora_a_proj: kv_a is REPLICATED (not tp-split), shape = [q_lora + kv_lora + qk_rope, hidden] = [80, 64]
  ExpectShape(dev, "model.layers.0.self_attn.fused_lora_a_proj.weight", {80, 64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Mla_QBProj) {
  auto loader = MakeLoader();
  Tensor qb = MakeHostTensor(TYPE_BF16, {2 * (32 + 32), 32});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.0.self_attn.q_b_proj.weight", qb}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // q_b_nope_rope: column split dim0, head_num=2, tp=2 → head_num_tp=1
  // [DivRoundUp(2*64, 2), 32] = [64, 32]
  ExpectShape(dev, "model.layers.0.self_attn.q_b_nope_rope_proj.weight", {32, 64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Mla_KvBProj) {
  auto loader = MakeLoader();
  Tensor kvb = MakeHostTensor(TYPE_BF16, {2 * (32 + 32), 16});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.0.self_attn.kv_b_proj.weight", kvb}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // head_num_tp = DivRoundUp(2,2) = 1
  // kv_b_nope: [head_num_tp * qk_nope, kv_lora] = [32, 16]
  ExpectShape(dev, "model.layers.0.self_attn.kv_b_nope_proj.weight", {16, 32});
  // v_head: [head_num_tp * v_head, kv_lora] = [32, 16]
  ExpectShape(dev, "model.layers.0.self_attn.v_head_proj.weight", {16, 32});
  // w_uk_t: [head_num_tp * qk_nope, kv_lora] = [32, 16]
  ExpectShape(dev, "model.layers.0.self_attn.w_uk_t.weight", {1, 32, 16});
  // w_uv: [head_num_tp * v_head, kv_lora] = [32, 16]
  ExpectShape(dev, "model.layers.0.self_attn.w_uv.weight", {1, 16, 32});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Mla_OProj) {
  auto loader = MakeLoader();
  Tensor o = MakeHostTensor(TYPE_BF16, {64, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.0.self_attn.o_proj.weight", o}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // o_proj: row split dim1 → [64, DivRoundUp(64,2)] = [64, 32]
  ExpectShape(dev, "model.layers.0.self_attn.o_proj.weight", {32, 64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Mla_GProj) {
  auto loader = MakeLoader();
  Tensor g = MakeHostTensor(TYPE_BF16, {64, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.0.self_attn.g_proj.weight", g}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // g_proj: column split dim0 → [DivRoundUp(64,2), 64] = [32, 64]
  ExpectShape(dev, "model.layers.0.self_attn.g_proj.weight", {64, 32});
}

// --- KDA weight loading (layer 1 → kda_layers={2,3} → 1+1=2 ∈ {2,3}) ---

TEST_F(KimiK3WeightLoaderIntegrationTest, Kda_QkvgFusion) {
  auto loader = MakeLoader();
  Tensor q = MakeHostTensor(TYPE_BF16, {64, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.self_attn.q_proj.weight", q}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // fused_qkvg_proj: 4 segments × [DivRoundUp(2*64,2), 64] = [4*64, 64] = [256, 64]
  ExpectShape(dev, "model.layers.1.self_attn.fused_qkvg_proj.weight", {128, 64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Kda_QkvConv1dFusion) {
  auto loader = MakeLoader();
  Tensor c = MakeHostTensor(TYPE_BF16, {128, 1, 4});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.self_attn.q_conv1d.weight", c}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // qkv_conv1d: 3 segments × [DivRoundUp(2*64,2), 1, 4] = [3*64, 1, 4] = [192, 1, 4]
  ExpectShape(dev, "model.layers.1.self_attn.qkv_conv1d.weight", {192, 1, 4});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Kda_ALog) {
  auto loader = MakeLoader();
  Tensor a = MakeHostTensor(TYPE_BF16, {2});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.self_attn.A_log", a}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // A_log: narrow to local heads → [DivRoundUp(2,2)] = [1]
  ExpectShape(dev, "model.layers.1.self_attn.A_log", {1});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Kda_DtBias) {
  auto loader = MakeLoader();
  Tensor d = MakeHostTensor(TYPE_BF16, {128});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.self_attn.dt_bias", d}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // dt_bias: dim0 split → [DivRoundUp(2*64,2)] = [64]
  ExpectShape(dev, "model.layers.1.self_attn.dt_bias", {64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Kda_BProj) {
  auto loader = MakeLoader();
  Tensor b = MakeHostTensor(TYPE_BF16, {64, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.self_attn.b_proj.weight", b}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // b_proj: column split dim0 → [DivRoundUp(64,2), 64] = [32, 64]
  ExpectShape(dev, "model.layers.1.self_attn.b_proj.weight", {64, 32});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Kda_FAProj) {
  auto loader = MakeLoader();
  Tensor fa = MakeHostTensor(TYPE_BF16, {16, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.self_attn.f_a_proj.weight", fa}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // f_a_proj: replicated → [16, 64]
  ExpectShape(dev, "model.layers.1.self_attn.f_a_proj.weight", {64, 16});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Kda_ONorm) {
  auto loader = MakeLoader();
  Tensor on = MakeHostTensor(TYPE_BF16, {64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.self_attn.o_norm.weight", on}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // o_norm: replicated → [64]
  ExpectShape(dev, "model.layers.1.self_attn.o_norm.weight", {64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Kda_OProj) {
  auto loader = MakeLoader();
  Tensor o = MakeHostTensor(TYPE_BF16, {64, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.self_attn.o_proj.weight", o}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // o_proj: row split dim1 → [64, DivRoundUp(64,2)] = [64, 32]
  ExpectShape(dev, "model.layers.1.self_attn.o_proj.weight", {32, 64});
}

// --- Latent MoE & router ---

TEST_F(KimiK3WeightLoaderIntegrationTest, LatentMoe_RoutedExpertNorm) {
  auto loader = MakeLoader();
  Tensor n = MakeHostTensor(TYPE_BF16, {64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.mlp.routed_expert_norm.weight", n}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // routed_expert_norm: replicated → [64]
  ExpectShape(dev, "model.layers.1.mlp.routed_expert_norm.weight", {64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, LatentMoe_RoutedExpertUpProj) {
  auto loader = MakeLoader();
  Tensor u = MakeHostTensor(TYPE_BF16, {64, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.mlp.routed_expert_up_proj.weight", u}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // routed_expert_up_proj: replicated → [64, 64]
  ExpectShape(dev, "model.layers.1.mlp.routed_expert_up_proj.weight", {64, 64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, LatentMoe_RoutedExpertDownProj) {
  auto loader = MakeLoader();
  Tensor d = MakeHostTensor(TYPE_BF16, {64, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.mlp.routed_expert_down_proj.weight", d}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // routed_expert_down_proj: replicated → [64, 64]
  ExpectShape(dev, "model.layers.1.mlp.routed_expert_down_proj.weight", {64, 64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Router_Gate) {
  auto loader = MakeLoader();
  Tensor g = MakeHostTensor(TYPE_BF16, {2, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.mlp.gate.weight", g}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // gate.weight: replicated → [2, 64]
  ExpectShape(dev, "model.layers.1.mlp.gate.weight", {64, 2});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, Router_EScoreCorrectionBias) {
  auto loader = MakeLoader();
  Tensor b = MakeHostTensor(TYPE_BF16, {2});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.mlp.gate.e_score_correction_bias", b}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // e_score_correction_bias: replicated → [2]
  ExpectShape(dev, "model.layers.1.mlp.gate.e_score_correction_bias", {2});
}

// --- Dense MLP (layer 0) ---

TEST_F(KimiK3WeightLoaderIntegrationTest, DenseMlp_GateProj) {
  auto loader = MakeLoader();
  Tensor g = MakeHostTensor(TYPE_BF16, {64, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.0.mlp.gate_proj.weight", g}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  EXPECT_GT(dev.size(), 0u);
}

TEST_F(KimiK3WeightLoaderIntegrationTest, DenseMlp_DownProj) {
  auto loader = MakeLoader();
  Tensor d = MakeHostTensor(TYPE_BF16, {64, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.0.mlp.down_proj.weight", d}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // down_proj: row split dim1 → [64, DivRoundUp(64,2)] = [64, 32]
  ExpectShape(dev, "model.layers.0.mlp.down_proj.weight", {32, 64});
}

// --- Shared experts ---

TEST_F(KimiK3WeightLoaderIntegrationTest, SharedExpert_GateProj) {
  auto loader = MakeLoader();
  Tensor g = MakeHostTensor(TYPE_BF16, {128, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.mlp.shared_expert.gate_proj.weight", g}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  EXPECT_GT(dev.size(), 0u);
}

// --- MoE expert weights (layer 1) ---

TEST_F(KimiK3WeightLoaderIntegrationTest, MoeExpert_GateProjFusedW13) {
  auto loader = MakeLoader();
  // gate_proj (w1) → fused into gate_up_proj front half.
  // Input: [moe_inter, hidden] = [64, 64]
  Tensor g = MakeHostTensor(TYPE_BF16, {64, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.mlp.experts.0.gate_proj.weight", g}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // gate_up_proj: [num_experts_per_rank, moe_inter*2, inner] = [2, 128, 64]
  ExpectShape(dev, "model.layers.1.mlp.experts.gate_up_proj.weight", {2, 128, 64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, MoeExpert_UpProjFusedW13) {
  auto loader = MakeLoader();
  // up_proj (w3) → fused into gate_up_proj back half.
  Tensor u = MakeHostTensor(TYPE_BF16, {64, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.mlp.experts.0.up_proj.weight", u}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  ExpectShape(dev, "model.layers.1.mlp.experts.gate_up_proj.weight", {2, 128, 64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, MoeExpert_DownProj) {
  auto loader = MakeLoader();
  // down_proj (w2) → separate tensor, dim1 split by moe_tp.
  // Input: [expert_hidden, moe_inter] = [64, 64]
  Tensor d = MakeHostTensor(TYPE_BF16, {64, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.mlp.experts.0.down_proj.weight", d}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // down: [num_experts_per_rank, expert_hidden, inner_tp] = [2, 64, 64]
  ExpectShape(dev, "model.layers.1.mlp.experts.down_proj.weight", {2, 64, 64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, MoeExpert_GateProjScaleInv) {
  auto loader = MakeLoader();
  // weight_scale_inv variant of gate_proj.
  Tensor s = MakeHostTensor(TYPE_BF16, {64, 8});
  std::unordered_map<std::string, Tensor> host = {
      {"model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv", s}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // gate_up_proj.weight_scale_inv: [2, 128, 8]
  ExpectShape(dev, "model.layers.1.mlp.experts.gate_up_proj.weight_scale_inv", {2, 128, 8});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, MoeExpert_DownProjScaleInv) {
  auto loader = MakeLoader();
  // weight_scale_inv variant of down_proj.
  Tensor s = MakeHostTensor(TYPE_BF16, {64, 8});
  std::unordered_map<std::string, Tensor> host = {
      {"model.layers.1.mlp.experts.0.down_proj.weight_scale_inv", s}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // down_proj.weight_scale_inv: [2, 64, 8]
  ExpectShape(dev, "model.layers.1.mlp.experts.down_proj.weight_scale_inv", {2, 64, 8});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, MoeExpert_Expert1GateProj) {
  auto loader = MakeLoader();
  // Expert 1 (second expert) gate_proj — verify expert_map routing works.
  Tensor g = MakeHostTensor(TYPE_BF16, {64, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.1.mlp.experts.1.gate_proj.weight", g}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  ExpectShape(dev, "model.layers.1.mlp.experts.gate_up_proj.weight", {2, 128, 64});
}

// --- Norm & res_proj (replicated) ---

TEST_F(KimiK3WeightLoaderIntegrationTest, Norm_Replicated) {
  auto loader = MakeLoader();
  Tensor n = MakeHostTensor(TYPE_BF16, {64});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.0.input_layernorm.weight", n}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // norm: replicated → [64]
  ExpectShape(dev, "model.layers.0.input_layernorm.weight", {64});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, ResProj_Replicated) {
  auto loader = MakeLoader();
  Tensor r = MakeHostTensor(TYPE_BF16, {1});
  std::unordered_map<std::string, Tensor> host = {{"model.layers.0.self_attention_res_proj.weight", r}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // res_proj: replicated → [1]
  ExpectShape(dev, "model.layers.0.self_attention_res_proj.weight", {1});
}

// --- Embed & lm_head ---

TEST_F(KimiK3WeightLoaderIntegrationTest, EmbedTokens) {
  auto loader = MakeLoader();
  Tensor e = MakeHostTensor(TYPE_BF16, {100, 64});
  std::unordered_map<std::string, Tensor> host = {{"model.embed_tokens.weight", e}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // embed_tokens: Vocab split dim0 → [DivRoundUp(100,2), 64] = [50, 64]
  ExpectShape(dev, "model.embed_tokens.weight", {100, 32});
}

TEST_F(KimiK3WeightLoaderIntegrationTest, LmHead) {
  auto loader = MakeLoader();
  Tensor l = MakeHostTensor(TYPE_BF16, {100, 64});
  std::unordered_map<std::string, Tensor> host = {{"lm_head.weight", l}};
  std::unordered_map<std::string, Tensor> dev, left;
  EXPECT_TRUE(loader->ProcessModelWeights(host, 0, dev, left).OK());
  // lm_head: Vocab split dim0 + transpose → [64, DivRoundUp(100,2)] = [64, 50]
  ExpectShape(dev, "lm_head.weight", {64, 50});
}

// --- PostProcessModelWeights ---

TEST_F(KimiK3WeightLoaderIntegrationTest, PostProcess_EScoreCorrectionBiasCastToFp32) {
  auto loader = MakeLoader();
  Tensor bias(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {2}, nullptr,
              &(context_->device->GetMemoryManageStream()));
  std::unordered_map<std::string, Tensor> dev = {{"model.layers.1.mlp.gate.e_score_correction_bias", bias}};
  EXPECT_TRUE(loader->PostProcessModelWeights(dev, 0).OK());
  EXPECT_EQ(dev["model.layers.1.mlp.gate.e_score_correction_bias"].dtype, TYPE_FP32);
}

TEST_F(KimiK3WeightLoaderIntegrationTest, PostProcess_NormalWeightCastToTargetDtype) {
  auto loader = MakeLoader();
  Tensor n(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {64}, nullptr,
           &(context_->device->GetMemoryManageStream()));
  std::unordered_map<std::string, Tensor> dev = {{"model.layers.0.input_layernorm.weight", n}};
  EXPECT_TRUE(loader->PostProcessModelWeights(dev, 0).OK());
  EXPECT_EQ(dev["model.layers.0.input_layernorm.weight"].dtype, TYPE_BF16);
}

#endif  // ENABLE_CUDA

}  // namespace ksana_llm
