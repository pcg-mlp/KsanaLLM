/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/kimi_k3/kimi_k3_config_parser.h"

#include <memory>
#include <string>

#include "models/kimi_k3/kimi_k3_config.h"
#include "nlohmann/json.hpp"
#include "test.h"
#include "utils/singleton.h"

using namespace ksana_llm;

class KimiK3ConfigParserTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    parser_ = std::make_unique<KimiK3ConfigParser>();
  }

  void TearDown() override { parser_.reset(); }

  // Base config based on Kimi-K3 defaults.
  nlohmann::json CreateBaseConfig() {
    return {{"torch_dtype", "bfloat16"},
            {"num_attention_heads", 96},
            {"num_key_value_heads", 96},
            {"intermediate_size", 33792},
            {"vocab_size", 163840},
            {"num_hidden_layers", 93},
            {"hidden_size", 7168},
            {"rope_theta", 10000.0},
            {"rms_norm_eps", 1e-5},
            {"bos_token_id", 163584},
            {"eos_token_id", 163586},
            {"pad_token_id", 163839},
            {"max_position_embeddings", 1048576},
            {"tie_word_embeddings", false},
            {"hidden_act", "situ"},
            {"activation_situ_beta", 4.0},
            {"activation_situ_linear_beta", 25.0},
            {"q_lora_rank", 1536},
            {"kv_lora_rank", 512},
            {"qk_nope_head_dim", 128},
            {"qk_rope_head_dim", 64},
            {"v_head_dim", 128},
            {"mla_use_nope", true},
            {"mla_use_output_gate", false},
            {"num_experts", 896},
            {"num_shared_experts", 2},
            {"moe_intermediate_size", 3072},
            {"num_experts_per_token", 16},
            {"first_k_dense_replace", 1},
            {"num_expert_group", 1},
            {"topk_group", 1},
            {"moe_router_activation_func", "sigmoid"},
            {"topk_method", "noaux_tc"},
            {"moe_renormalize", true},
            {"routed_scaling_factor", 1.0},
            {"routed_expert_hidden_size", 3584},
            {"latent_moe_use_norm", true},
            {"attn_res_block_size", 128},
            {"linear_attn_config",
             {{"kda_layers", {2, 3, 5, 7}},
              {"num_heads", 96},
              {"head_dim", 128},
              {"short_conv_kernel_size", 4},
              {"gate_lower_bound", -5.0},
              {"use_full_rank_gate", true}}}};
  }

  // Parse config and return KimiK3Config.
  std::shared_ptr<KimiK3Config> ParseConfig(const nlohmann::json& config_json) {
    ParallelismBasicConfig parallel_basic_config;
    parallel_basic_config.tensor_parallel_size = 1;
    std::shared_ptr<BaseModelConfig> model_config;
    Status status = parser_->ParseModelConfig(config_json, parallel_basic_config, "", model_config);
    EXPECT_TRUE(status.OK());
    return std::dynamic_pointer_cast<KimiK3Config>(model_config);
  }

  std::unique_ptr<KimiK3ConfigParser> parser_;
};

// Test basic model config parsing with full config.
TEST_F(KimiK3ConfigParserTest, ParseModelConfigBasic) {
  auto cfg = ParseConfig(CreateBaseConfig());
  ASSERT_NE(cfg, nullptr);

  // Common attributes.
  EXPECT_EQ(cfg->head_num, 96u);
  EXPECT_EQ(cfg->num_key_value_heads, 96u);
  EXPECT_EQ(cfg->inter_size, 33792u);
  EXPECT_EQ(cfg->vocab_size, 163840u);
  EXPECT_EQ(cfg->num_layer, 93u);
  EXPECT_EQ(cfg->hidden_units, 7168u);
  EXPECT_FLOAT_EQ(cfg->rope_theta, 10000.0f);
  EXPECT_FLOAT_EQ(cfg->layernorm_eps, 1e-5f);
  EXPECT_EQ(cfg->start_id, 163584u);
  EXPECT_EQ(cfg->end_ids, std::vector<uint32_t>{163586u});
  EXPECT_EQ(cfg->pad_id, 163839u);
  EXPECT_EQ(cfg->max_position_embeddings, 1048576);
  EXPECT_FALSE(cfg->tie_word_embeddings);
  EXPECT_TRUE(cfg->exist_tie_embeddings_param);
  EXPECT_EQ(cfg->weight_data_type, TYPE_BF16);

  // Activation (situ).
  EXPECT_EQ(cfg->activation_function, "situ");
  EXPECT_FLOAT_EQ(cfg->activation_situ_beta, 4.0f);
  EXPECT_FLOAT_EQ(cfg->activation_situ_linear_beta, 25.0f);

  // MLA config.
  EXPECT_TRUE(cfg->use_mla);
  EXPECT_EQ(cfg->mla_config.q_lora_rank, 1536u);
  EXPECT_EQ(cfg->mla_config.kv_lora_rank, 512u);
  EXPECT_EQ(cfg->mla_config.qk_nope_head_dim, 128u);
  EXPECT_EQ(cfg->mla_config.qk_rope_head_dim, 64u);
  EXPECT_EQ(cfg->mla_config.v_head_dim, 128u);
  EXPECT_TRUE(cfg->mla_config.mla_use_nope);
  EXPECT_FALSE(cfg->mla_config.mla_use_output_gate);
  EXPECT_EQ(cfg->size_per_head, 192u);  // 128 + 64
  EXPECT_EQ(cfg->rotary_embedding, 64u);

  // MoE config.
  EXPECT_TRUE(cfg->is_moe);
  EXPECT_TRUE(cfg->moe_config.use_vllm_moe);
  EXPECT_EQ(cfg->moe_config.num_experts, 896u);
  EXPECT_EQ(cfg->moe_config.moe_inter_size, 3072u);
  EXPECT_EQ(cfg->moe_config.experts_topk, 16u);
  EXPECT_EQ(cfg->moe_config.first_k_dense_replace, 1u);
  EXPECT_EQ(cfg->moe_config.scoring_func, "sigmoid");
  EXPECT_EQ(cfg->moe_config.topk_method, "noaux_tc");
  EXPECT_TRUE(cfg->moe_config.norm_topk_prob);
  EXPECT_TRUE(cfg->moe_config.use_e_score_correction_bias);
  EXPECT_EQ(cfg->moe_config.num_shared_experts, 2u);
  EXPECT_TRUE(cfg->has_shared_experts);
  EXPECT_EQ(cfg->moe_config.shared_expert_inter_size, 6144u);  // 2 * 3072
  EXPECT_EQ(cfg->moe_config.routed_expert_hidden_size, 3584u);
  EXPECT_TRUE(cfg->moe_config.latent_moe_use_norm);

  // Kimi Linear (KDA) config.
  EXPECT_EQ(cfg->kimi_linear_config.kda_layers.size(), 4u);
  EXPECT_EQ(cfg->kimi_linear_config.short_conv_kernel_size, 4u);
  EXPECT_EQ(cfg->kimi_linear_config.head_dim, 128u);
  EXPECT_FLOAT_EQ(cfg->kimi_linear_config.gate_lower_bound, -5.0f);
  EXPECT_TRUE(cfg->kimi_linear_config.use_full_rank_gate);

  // AttnRes config.
  EXPECT_EQ(cfg->attn_res_config.attn_res_block_size, 128u);

  // Parallelism.
  EXPECT_EQ(cfg->tensor_para_size, 1);
  EXPECT_EQ(cfg->expert_para_size, 1);
  EXPECT_EQ(cfg->moe_tensor_para_size, 1);
}

// Test default values with minimal config.
TEST_F(KimiK3ConfigParserTest, ParseModelConfigDefaults) {
  nlohmann::json minimal_config = {{"torch_dtype", "bfloat16"}};

  auto cfg = ParseConfig(minimal_config);
  ASSERT_NE(cfg, nullptr);

  // Kimi-K3 specific defaults.
  EXPECT_EQ(cfg->head_num, 96u);
  EXPECT_EQ(cfg->inter_size, 33792u);
  EXPECT_EQ(cfg->vocab_size, 163840u);
  EXPECT_EQ(cfg->num_layer, 93u);
  EXPECT_EQ(cfg->hidden_units, 7168u);
  EXPECT_FLOAT_EQ(cfg->rope_theta, 10000.0f);
  EXPECT_FLOAT_EQ(cfg->layernorm_eps, 1e-5f);
  EXPECT_EQ(cfg->start_id, 163584u);
  EXPECT_EQ(cfg->end_ids, std::vector<uint32_t>{163586u});
  EXPECT_EQ(cfg->pad_id, 163839u);
  EXPECT_EQ(cfg->max_position_embeddings, 1048576);
  EXPECT_FALSE(cfg->exist_tie_embeddings_param);
  EXPECT_FALSE(cfg->tie_word_embeddings);

  // Activation defaults.
  EXPECT_EQ(cfg->activation_function, "situ");
  EXPECT_FLOAT_EQ(cfg->activation_situ_beta, 4.0f);
  EXPECT_FLOAT_EQ(cfg->activation_situ_linear_beta, 25.0f);

  // MLA defaults (q_lora_rank not present → 0).
  EXPECT_TRUE(cfg->use_mla);
  EXPECT_EQ(cfg->mla_config.q_lora_rank, 0u);
  EXPECT_EQ(cfg->mla_config.kv_lora_rank, 512u);
  EXPECT_EQ(cfg->size_per_head, 192u);  // 128 + 64

  // MoE defaults.
  EXPECT_TRUE(cfg->is_moe);
  EXPECT_EQ(cfg->moe_config.num_experts, 896u);
  EXPECT_EQ(cfg->moe_config.moe_inter_size, 3072u);
  EXPECT_EQ(cfg->moe_config.experts_topk, 16u);
  EXPECT_EQ(cfg->moe_config.num_shared_experts, 2u);
  EXPECT_TRUE(cfg->has_shared_experts);

  // No linear_attn_config → empty kda_layers.
  EXPECT_TRUE(cfg->kimi_linear_config.kda_layers.empty());
  EXPECT_EQ(cfg->attn_res_config.attn_res_block_size, 0u);
}

// Test eos_token_id as array.
TEST_F(KimiK3ConfigParserTest, ParseModelConfigEosTokenIdArray) {
  nlohmann::json config = CreateBaseConfig();
  config["eos_token_id"] = {163586, 163587};

  auto cfg = ParseConfig(config);
  ASSERT_NE(cfg, nullptr);
  EXPECT_EQ(cfg->end_ids, (std::vector<uint32_t>{163586u, 163587u}));
}
