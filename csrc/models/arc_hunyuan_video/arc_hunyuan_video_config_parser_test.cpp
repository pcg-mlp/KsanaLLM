/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/arc_hunyuan_video/arc_hunyuan_video_config_parser.h"

#include <memory>
#include <string>

#include "models/arc_hunyuan_video/arc_hunyuan_video_config.h"
#include "nlohmann/json.hpp"
#include "test.h"
#include "utils/singleton.h"

using namespace ksana_llm;

class ArcHunyuanVideoConfigParserTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    parser_ = std::make_unique<ArcHunyuanVideoConfigParser>();
  }

  void TearDown() override { parser_.reset(); }

  // Base config from ARC-Hunyuan-Video-7B
  nlohmann::json CreateBaseConfig() {
    return {{"torch_dtype", "bfloat16"},
            {"text_config",
             {{"num_attention_heads", 32},
              {"num_key_value_heads", 8},
              {"intermediate_size", 14336},
              {"vocab_size", 128256},
              {"num_hidden_layers", 32},
              {"hidden_size", 4096},
              {"rope_theta", 10000.0},
              {"rms_norm_eps", 1e-5},
              {"bos_token_id", 127959},
              {"eos_token_id", 127960},
              {"pad_token_id", 127961},
              {"max_position_embeddings", 1024},
              {"attention_head_dim", 128},
              {"hidden_act", "silu"},
              {"attention_bias", false},
              {"attention_dropout", 0.0},
              {"eod_token_id", 127957},
              {"im_end_id", 127963},
              {"im_newline_id", 127964},
              {"im_start_id", 127962},
              {"image_token_id", 127968},
              {"initializer_range", 0.02},
              {"is_causal", true},
              {"mlp_bias", false},
              {"norm_type", "hf_rms"},
              {"num_media_embeds", 257},
              {"position_embedding_xdrope", true},
              {"use_qk_norm", true},
              {"use_rotary_pos_emb", true},
              {"rope_scaling",
               {{"type", "dynamic"},
                {"factor", 1.0},
                {"alpha", 1000.0},
                {"beta_fast", 32.0},
                {"beta_slow", 1.0},
                {"mscale", 1.0},
                {"mscale_all_dim", 1.0}}},
              {"xdrope_section", {0.25, 0.25, 0.25, 0.25}}}},
            {"tie_word_embeddings", false}};
  }

  // Parse config and return ArcHunyuanVideoConfig
  std::shared_ptr<ArcHunyuanVideoConfig> ParseConfig(const nlohmann::json& config_json) {
    ParallelismBasicConfig parallel_basic_config;
    parallel_basic_config.tensor_parallel_size = 1;
    std::shared_ptr<BaseModelConfig> model_config;
    Status status = parser_->ParseModelConfig(config_json, parallel_basic_config, "", model_config);
    EXPECT_TRUE(status.OK());
    return std::dynamic_pointer_cast<ArcHunyuanVideoConfig>(model_config);
  }

  std::unique_ptr<ArcHunyuanVideoConfigParser> parser_;
};

// Test basic config parsing with/without quantization
TEST_F(ArcHunyuanVideoConfigParserTest, ParseModelConfigBasic) {
  // fp8 quantization
  nlohmann::json config_with_quant = CreateBaseConfig();
  config_with_quant["quantization_config"] = {{"quant_method", "fp8"}, {"activation_scheme", "static"}};

  auto arc_config = ParseConfig(config_with_quant);
  ASSERT_NE(arc_config, nullptr);

  // Verify text_config core parameters
  EXPECT_EQ(arc_config->head_num, 32);
  EXPECT_EQ(arc_config->num_key_value_heads, 8);
  EXPECT_EQ(arc_config->inter_size, 14336);
  EXPECT_EQ(arc_config->vocab_size, 128256);
  EXPECT_EQ(arc_config->num_layer, 32);
  EXPECT_EQ(arc_config->hidden_units, 4096);
  EXPECT_FLOAT_EQ(arc_config->rope_theta, 10000.0f);
  EXPECT_FLOAT_EQ(arc_config->layernorm_eps, 1e-5f);
  EXPECT_EQ(arc_config->start_id, 127959);
  EXPECT_EQ(arc_config->end_id, 127960);
  EXPECT_EQ(arc_config->pad_id, 127961);
  EXPECT_EQ(arc_config->max_position_embeddings, 1024);
  EXPECT_EQ(arc_config->size_per_head, 128);
  EXPECT_EQ(arc_config->activation_function, "silu");
  EXPECT_FALSE(arc_config->tie_word_embeddings);

  // Verify quant fields
  EXPECT_TRUE(arc_config->is_quant);
  EXPECT_EQ(arc_config->quant_config.method, QUANT_FP8_E4M3);
  EXPECT_TRUE(arc_config->quant_config.is_checkpoint_fp8_serialized);
  EXPECT_TRUE(arc_config->quant_config.is_activation_scheme_static);

  // Without quantization
  nlohmann::json config_no_quant = CreateBaseConfig();
  auto arc_config_no_quant = ParseConfig(config_no_quant);
  EXPECT_FALSE(arc_config_no_quant->is_quant);
}

// Test special field handling: missing text_config, tie_word_embeddings, head_groups
TEST_F(ArcHunyuanVideoConfigParserTest, ParseModelConfigSpecialFields) {
  // Missing text_config should throw
  nlohmann::json config_no_text = {{"torch_dtype", "bfloat16"}};
  ParallelismBasicConfig parallel_basic_config;
  std::shared_ptr<BaseModelConfig> model_config;
  EXPECT_THROW(parser_->ParseModelConfig(config_no_text, parallel_basic_config, "", model_config), std::runtime_error);

  // tie_word_embeddings=true
  nlohmann::json config = CreateBaseConfig();
  config["tie_word_embeddings"] = true;
  auto arc_config = ParseConfig(config);
  EXPECT_TRUE(arc_config->tie_word_embeddings);
  EXPECT_TRUE(arc_config->exist_tie_embeddings_param);

  // tie_word_embeddings absent
  config = CreateBaseConfig();
  config.erase("tie_word_embeddings");
  arc_config = ParseConfig(config);
  EXPECT_FALSE(arc_config->exist_tie_embeddings_param);

  // head_groups = num_attention_heads / num_key_value_heads
  config = CreateBaseConfig();
  config["text_config"]["num_attention_heads"] = 64;
  config["text_config"]["num_key_value_heads"] = 8;
  arc_config = ParseConfig(config);
  EXPECT_EQ(arc_config->head_groups, 8);
}

// Test text_config fields
TEST_F(ArcHunyuanVideoConfigParserTest, ParseTextConfigSpecificParams) {
  nlohmann::json config = CreateBaseConfig();
  config["text_config"]["attention_bias"] = true;
  config["text_config"]["attention_dropout"] = 0.1;
  config["text_config"]["eod_token_id"] = 200;
  config["text_config"]["im_end_id"] = 201;
  config["text_config"]["im_newline_id"] = 202;
  config["text_config"]["im_start_id"] = 203;
  config["text_config"]["image_token_id"] = 204;
  config["text_config"]["initializer_range"] = 0.05;
  config["text_config"]["is_causal"] = false;
  config["text_config"]["mlp_bias"] = true;
  config["text_config"]["norm_type"] = "layer_norm";
  config["text_config"]["num_media_embeds"] = 512;
  config["text_config"]["position_embedding_xdrope"] = false;
  config["text_config"]["use_qk_norm"] = false;
  config["text_config"]["use_rotary_pos_emb"] = false;

  auto arc_config = ParseConfig(config);
  ASSERT_NE(arc_config, nullptr);

  EXPECT_TRUE(arc_config->attention_bias);
  EXPECT_FLOAT_EQ(arc_config->attention_dropout, 0.1f);
  EXPECT_EQ(arc_config->eod_token_id, 200);
  EXPECT_EQ(arc_config->im_end_id, 201);
  EXPECT_EQ(arc_config->im_newline_id, 202);
  EXPECT_EQ(arc_config->im_start_id, 203);
  EXPECT_EQ(arc_config->image_token_id, 204);
  EXPECT_FLOAT_EQ(arc_config->initializer_range, 0.05f);
  EXPECT_FALSE(arc_config->is_causal);
  EXPECT_TRUE(arc_config->mlp_bias);
  EXPECT_EQ(arc_config->norm_type, "layer_norm");
  EXPECT_EQ(arc_config->num_media_embeds, 512);
  EXPECT_FALSE(arc_config->position_embedding_xdrope);
  EXPECT_FALSE(arc_config->use_qk_norm);
  EXPECT_FALSE(arc_config->use_rotary_pos_emb);
}

// Test rope_scaling parsing
TEST_F(ArcHunyuanVideoConfigParserTest, ParseRopeScaling) {
  // Custom rope_scaling values
  nlohmann::json config = CreateBaseConfig();
  config["text_config"]["rope_scaling"] = {{"type", "yarn"},       {"factor", 2.5},    {"alpha", 500.0},
                                           {"beta_fast", 16.0},    {"beta_slow", 2.0}, {"mscale", 0.5},
                                           {"mscale_all_dim", 0.8}};

  auto arc_config = ParseConfig(config);
  ASSERT_NE(arc_config, nullptr);

  EXPECT_EQ(arc_config->rope_scaling_factor_config.type, "yarn");
  EXPECT_FLOAT_EQ(arc_config->rope_scaling_factor_config.factor, 2.5f);
  EXPECT_FLOAT_EQ(arc_config->rope_scaling_factor_config.scaling_alpha, 500.0f);
  EXPECT_FLOAT_EQ(arc_config->rope_scaling_factor_config.beta_fast, 16.0f);
  EXPECT_FLOAT_EQ(arc_config->rope_scaling_factor_config.beta_slow, 2.0f);
  EXPECT_FLOAT_EQ(arc_config->rope_scaling_factor_config.mscale, 0.5f);
  EXPECT_FLOAT_EQ(arc_config->rope_scaling_factor_config.mscale_all_dim, 0.8f);

  // Empty rope_scaling uses default values
  config = CreateBaseConfig();
  config["text_config"]["rope_scaling"] = nlohmann::json::object();
  arc_config = ParseConfig(config);
  EXPECT_EQ(arc_config->rope_scaling_factor_config.type, "dynamic");
  EXPECT_FLOAT_EQ(arc_config->rope_scaling_factor_config.factor, 1.0f);
  EXPECT_FLOAT_EQ(arc_config->rope_scaling_factor_config.scaling_alpha, 1000.0f);
  EXPECT_FLOAT_EQ(arc_config->rope_scaling_factor_config.beta_fast, 32.0f);
  EXPECT_FLOAT_EQ(arc_config->rope_scaling_factor_config.beta_slow, 1.0f);
  EXPECT_FLOAT_EQ(arc_config->rope_scaling_factor_config.mscale, 1.0f);
  EXPECT_FLOAT_EQ(arc_config->rope_scaling_factor_config.mscale_all_dim, 1.0f);
}

// Test xdrope_section float-to-int conversion and cumsum
TEST_F(ArcHunyuanVideoConfigParserTest, ParseXdropeSection) {
  // size_per_head=128, factor=64: int(0.25*64)=16 each, cumsum=[16,32,48,64]
  nlohmann::json config = CreateBaseConfig();
  config["text_config"]["xdrope_section"] = {0.25, 0.25, 0.25, 0.25};
  auto arc_config = ParseConfig(config);
  ASSERT_NE(arc_config, nullptr);
  std::vector<int> expected = {16, 32, 48, 64};
  EXPECT_EQ(arc_config->rope_scaling_factor_config.xdrope_section, expected);

  // size_per_head=64, factor=32: int(0.5*32)=16 each, cumsum=[16,32]
  config = CreateBaseConfig();
  config["text_config"]["attention_head_dim"] = 64;
  config["text_config"]["xdrope_section"] = {0.5, 0.5};
  arc_config = ParseConfig(config);
  std::vector<int> expected2 = {16, 32};
  EXPECT_EQ(arc_config->rope_scaling_factor_config.xdrope_section, expected2);

  // Single element: size_per_head=128, int(1.0*64)=64, cumsum=[64]
  config = CreateBaseConfig();
  config["text_config"]["xdrope_section"] = {1.0};
  arc_config = ParseConfig(config);
  std::vector<int> expected3 = {64};
  EXPECT_EQ(arc_config->rope_scaling_factor_config.xdrope_section, expected3);
}

// Test quantization config parsing
TEST_F(ArcHunyuanVideoConfigParserTest, ParseQuantConfig) {
  // fp8 static activation scheme
  nlohmann::json config = CreateBaseConfig();
  config["quantization_config"] = {{"quant_method", "fp8"}, {"activation_scheme", "static"}};
  auto arc_config = ParseConfig(config);
  EXPECT_TRUE(arc_config->is_quant);
  EXPECT_EQ(arc_config->quant_config.method, QUANT_FP8_E4M3);
  EXPECT_TRUE(arc_config->quant_config.is_checkpoint_fp8_serialized);
  EXPECT_TRUE(arc_config->quant_config.is_activation_scheme_static);

  // fp8 without activation_scheme defaults to non-static
  config = CreateBaseConfig();
  config["quantization_config"] = {{"quant_method", "fp8"}};
  arc_config = ParseConfig(config);
  EXPECT_TRUE(arc_config->is_quant);
  EXPECT_EQ(arc_config->quant_config.method, QUANT_FP8_E4M3);
  EXPECT_TRUE(arc_config->quant_config.is_checkpoint_fp8_serialized);
  EXPECT_FALSE(arc_config->quant_config.is_activation_scheme_static);

  // fp8 with non-static activation_scheme
  config = CreateBaseConfig();
  config["quantization_config"] = {{"quant_method", "fp8"}, {"activation_scheme", "dynamic"}};
  arc_config = ParseConfig(config);
  EXPECT_TRUE(arc_config->is_quant);
  EXPECT_FALSE(arc_config->quant_config.is_activation_scheme_static);

  // No quantization config
  config = CreateBaseConfig();
  arc_config = ParseConfig(config);
  EXPECT_FALSE(arc_config->is_quant);

  // Unsupported quant_method should throw
  config = CreateBaseConfig();
  config["quantization_config"] = {{"quant_method", "gptq"}};
  ParallelismBasicConfig parallel_basic_config;
  std::shared_ptr<BaseModelConfig> model_config;
  EXPECT_THROW(parser_->ParseModelConfig(config, parallel_basic_config, "", model_config), std::runtime_error);

  // Missing quant_method field should throw
  config = CreateBaseConfig();
  config["quantization_config"] = {{"some_field", "value"}};
  EXPECT_THROW(parser_->ParseModelConfig(config, parallel_basic_config, "", model_config), std::runtime_error);
}

// Test default values and comprehensive config
TEST_F(ArcHunyuanVideoConfigParserTest, ParseModelConfigDefaultsAndComprehensive) {
  // Minimal config (use defaults)
  nlohmann::json minimal_config = {
      {"torch_dtype", "bfloat16"},
      {"text_config", {{"rope_scaling", {{"type", "dynamic"}}}, {"xdrope_section", {0.5, 0.5}}}}};

  auto arc_config = ParseConfig(minimal_config);
  EXPECT_EQ(arc_config->head_num, 32);
  EXPECT_EQ(arc_config->num_key_value_heads, 8);
  EXPECT_EQ(arc_config->vocab_size, 128256);
  EXPECT_FLOAT_EQ(arc_config->rope_theta, 10000.0f);
  EXPECT_EQ(arc_config->activation_function, "silu");
  EXPECT_TRUE(arc_config->is_causal);
  EXPECT_TRUE(arc_config->use_qk_norm);
  EXPECT_TRUE(arc_config->use_rotary_pos_emb);
  EXPECT_EQ(arc_config->head_groups, 4);

  // Comprehensive config with all custom values
  nlohmann::json comprehensive_config = CreateBaseConfig();
  comprehensive_config["torch_dtype"] = "float16";
  comprehensive_config["text_config"]["num_attention_heads"] = 64;
  comprehensive_config["text_config"]["num_key_value_heads"] = 16;
  comprehensive_config["text_config"]["intermediate_size"] = 28672;
  comprehensive_config["text_config"]["vocab_size"] = 256000;
  comprehensive_config["text_config"]["num_hidden_layers"] = 48;
  comprehensive_config["text_config"]["hidden_size"] = 8192;
  comprehensive_config["text_config"]["rope_theta"] = 500000.0;
  comprehensive_config["text_config"]["rms_norm_eps"] = 1e-6;
  comprehensive_config["text_config"]["bos_token_id"] = 100;
  comprehensive_config["text_config"]["eos_token_id"] = 101;
  comprehensive_config["text_config"]["pad_token_id"] = 102;
  comprehensive_config["text_config"]["max_position_embeddings"] = 4096;
  comprehensive_config["text_config"]["attention_head_dim"] = 256;
  comprehensive_config["text_config"]["hidden_act"] = "gelu";
  comprehensive_config["tie_word_embeddings"] = true;
  comprehensive_config["quantization_config"] = {{"quant_method", "fp8"}, {"activation_scheme", "static"}};

  arc_config = ParseConfig(comprehensive_config);
  EXPECT_EQ(arc_config->start_id, 100);
  EXPECT_EQ(arc_config->end_id, 101);
  EXPECT_EQ(arc_config->pad_id, 102);
  EXPECT_EQ(arc_config->size_per_head, 256);
  EXPECT_EQ(arc_config->hidden_units, 8192);
  EXPECT_EQ(arc_config->inter_size, 28672);
  EXPECT_EQ(arc_config->max_position_embeddings, 4096);
  EXPECT_EQ(arc_config->head_num, 64);
  EXPECT_EQ(arc_config->num_layer, 48);
  EXPECT_EQ(arc_config->num_key_value_heads, 16);
  EXPECT_FLOAT_EQ(arc_config->layernorm_eps, 1e-6f);
  EXPECT_FLOAT_EQ(arc_config->rope_theta, 500000.0f);
  EXPECT_TRUE(arc_config->tie_word_embeddings);
  EXPECT_EQ(arc_config->vocab_size, 256000);
  EXPECT_EQ(arc_config->activation_function, "gelu");
  EXPECT_EQ(arc_config->head_groups, 4);
  EXPECT_EQ(arc_config->weight_data_type, TYPE_FP16);
  EXPECT_TRUE(arc_config->is_quant);
  EXPECT_EQ(arc_config->quant_config.method, QUANT_FP8_E4M3);
}
