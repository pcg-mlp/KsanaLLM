/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "configure/quant_config_parser.h"
#include <gtest/gtest.h>
#include "configure/model_config.h"
#include "configure/model_config_parser.h"
#include "models/deepseek_v3/deepseek_v3_config.h"
#include "models/deepseek_v3/deepseek_v3_config_parser.h"

namespace ksana_llm {

TEST(QuantConfigParser, TestParseAndConvertQuantConfig) {
  // 构建数据
  std::string json_str = R"(
    {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": null
        }
    })";
  nlohmann::json hf_quant_config = nlohmann::json::parse(json_str);

  // 解析
  nlohmann::json quantization_config = ParseAndConvertQuantConfig(hf_quant_config);

  // 校验
  EXPECT_EQ(quantization_config["quant_method"], "mixed");
  EXPECT_EQ(quantization_config["configs"]["fp8"]["method"], "fp8");
  EXPECT_EQ(quantization_config["configs"]["w4a8"]["method"], "rtn");
  EXPECT_EQ(quantization_config["layer_mapping"]["fp8"]["default_config"], true);
  EXPECT_EQ(quantization_config["layer_mapping"]["w4a8"]["default_config"], false);
}

// ============================================================================
// Tests for compressed-tensors config parsing
// These tests call the REAL production code paths:
//   - EnvModelConfigParser::ParseModelQuantConfig (model_config_parser.cpp L112-142)
//   - DeepSeekV3ConfigParser::ParseQuantConfig (deepseek_v3_config_parser.cpp L164-201)
// ============================================================================

// Helper: build a Kimi-K2.5 style compressed-tensors quantization_config JSON
static nlohmann::json BuildCompressedTensorsConfig(int num_bits = 4, int group_size = 32,
                                                   bool with_ignore = true) {
  nlohmann::json config;
  config["quantization_config"] = {
      {"quant_method", "compressed-tensors"},
      {"config_groups",
       {{"group_0",
         {{"weights", {{"num_bits", num_bits}, {"group_size", group_size}, {"strategy", "group"}, {"type", "int"}}},
          {"targets", {"Linear"}}}}}}};
  if (with_ignore) {
    config["quantization_config"]["ignore"] = {"lm_head", "re:.*embed_tokens"};
  }
  return config;
}

// ---------------------------------------------------------------------------
// EnvModelConfigParser tests (model_config_parser.cpp compressed-tensors path)
// ---------------------------------------------------------------------------

// Test: basic compressed-tensors parsing via EnvModelConfigParser::ParseModelQuantConfig
TEST(QuantConfigParser, EnvPath_CompressedTensorsBasic) {
  nlohmann::json config_json = BuildCompressedTensorsConfig(4, 32, /*with_ignore=*/false);

  EnvModelConfigParser parser("auto", "cutlass");
  ModelConfig model_config;
  model_config.is_moe = true;
  model_config.path = "/tmp/nonexistent_model_dir";  // won't be read since config_json has quantization_config
  std::string yaml_quant = "auto";
  std::string yaml_gptq = "cutlass";

  parser.ParseModelQuantConfig(config_json, model_config, yaml_quant, yaml_gptq);

  EXPECT_TRUE(model_config.is_quant);
  EXPECT_EQ(model_config.quant_config.method, QUANT_GPTQ);
  EXPECT_EQ(model_config.quant_config.bits, 4u);
  EXPECT_EQ(model_config.quant_config.group_size, 32u);
  EXPECT_FALSE(model_config.quant_config.desc_act);
  EXPECT_FALSE(model_config.quant_config.input_scale);
  // Without ignore list, MoE INT4 should NOT be triggered
  EXPECT_FALSE(model_config.quant_config.enable_moe_int4);
  EXPECT_TRUE(model_config.sub_quant_configs.empty());
}

// Test: compressed-tensors with ignore list triggers MoE INT4 mode
TEST(QuantConfigParser, EnvPath_CompressedTensorsWithIgnoreList) {
  nlohmann::json config_json = BuildCompressedTensorsConfig(4, 32, /*with_ignore=*/true);

  EnvModelConfigParser parser("auto", "cutlass");
  ModelConfig model_config;
  model_config.is_moe = true;
  model_config.path = "/tmp/nonexistent_model_dir";
  std::string yaml_quant = "auto";
  std::string yaml_gptq = "cutlass";

  parser.ParseModelQuantConfig(config_json, model_config, yaml_quant, yaml_gptq);

  // Main quant_config should be reset with enable_moe_int4, but bits/group_size preserved
  EXPECT_TRUE(model_config.quant_config.enable_moe_int4);
  EXPECT_EQ(model_config.quant_config.method, QUANT_NONE);  // reset
  EXPECT_EQ(model_config.quant_config.bits, 4u);             // preserved
  EXPECT_EQ(model_config.quant_config.group_size, 32u);      // preserved

  // sub_quant_configs should have expert config
  ASSERT_EQ(model_config.sub_quant_configs.size(), 1u);
  EXPECT_EQ(model_config.sub_quant_configs[0].method, QUANT_GPTQ);
  EXPECT_EQ(model_config.sub_quant_configs[0].bits, 4u);
  EXPECT_EQ(model_config.sub_quant_configs[0].group_size, 32u);
  ASSERT_EQ(model_config.sub_quant_configs[0].pattern_layers.size(), 1u);
  EXPECT_EQ(model_config.sub_quant_configs[0].pattern_layers[0], ".mlp.experts.");
}

// Test: custom num_bits and group_size via env path
TEST(QuantConfigParser, EnvPath_CompressedTensorsCustomBitsGroupSize) {
  nlohmann::json config_json = BuildCompressedTensorsConfig(8, 64, /*with_ignore=*/false);

  EnvModelConfigParser parser("auto", "cutlass");
  ModelConfig model_config;
  model_config.is_moe = false;
  model_config.path = "/tmp/nonexistent_model_dir";
  std::string yaml_quant = "auto";
  std::string yaml_gptq = "cutlass";

  parser.ParseModelQuantConfig(config_json, model_config, yaml_quant, yaml_gptq);

  EXPECT_EQ(model_config.quant_config.bits, 8u);
  EXPECT_EQ(model_config.quant_config.group_size, 64u);
}

// ---------------------------------------------------------------------------
// DeepSeekV3ConfigParser tests (deepseek_v3_config_parser.cpp compressed-tensors path)
// ---------------------------------------------------------------------------

// Test: basic compressed-tensors parsing via DeepSeekV3ConfigParser::ParseQuantConfig
TEST(QuantConfigParser, DeepSeekV3Path_CompressedTensorsBasic) {
  nlohmann::json config_json = BuildCompressedTensorsConfig(4, 32, /*with_ignore=*/false);

  DeepSeekV3ConfigParser parser;
  auto ds_config = std::make_shared<DeepSeekV3Config>();
  ds_config->is_moe = true;
  ds_config->model_dir = "/tmp/nonexistent_model_dir";
  std::string yaml_quant = "auto";
  std::string yaml_gptq = "cutlass";

  Status status = parser.ParseQuantConfig(config_json, ds_config, yaml_quant, yaml_gptq);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  EXPECT_TRUE(ds_config->is_quant);
  EXPECT_TRUE(ds_config->is_compressed_tensors);
  EXPECT_EQ(ds_config->quant_config.method, QUANT_GPTQ);
  EXPECT_EQ(ds_config->quant_config.bits, 4u);
  EXPECT_EQ(ds_config->quant_config.group_size, 32u);
  EXPECT_TRUE(ds_config->quant_config.is_compressed_tensors);
}

// Test: compressed-tensors with ignore list via DeepSeekV3 path
TEST(QuantConfigParser, DeepSeekV3Path_CompressedTensorsWithIgnoreList) {
  nlohmann::json config_json = BuildCompressedTensorsConfig(4, 32, /*with_ignore=*/true);

  DeepSeekV3ConfigParser parser;
  auto ds_config = std::make_shared<DeepSeekV3Config>();
  ds_config->is_moe = true;
  ds_config->model_dir = "/tmp/nonexistent_model_dir";
  std::string yaml_quant = "auto";
  std::string yaml_gptq = "cutlass";

  Status status = parser.ParseQuantConfig(config_json, ds_config, yaml_quant, yaml_gptq);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  EXPECT_TRUE(ds_config->is_compressed_tensors);
  EXPECT_TRUE(ds_config->quant_config.enable_moe_int4);
  EXPECT_TRUE(ds_config->quant_config.is_compressed_tensors);
  EXPECT_EQ(ds_config->quant_config.method, QUANT_NONE);
  EXPECT_EQ(ds_config->quant_config.bits, 4u);
  EXPECT_EQ(ds_config->quant_config.group_size, 32u);

  ASSERT_EQ(ds_config->sub_quant_configs.size(), 1u);
  EXPECT_EQ(ds_config->sub_quant_configs[0].method, QUANT_GPTQ);
  EXPECT_TRUE(ds_config->sub_quant_configs[0].is_compressed_tensors);
  EXPECT_EQ(ds_config->sub_quant_configs[0].pattern_layers[0], ".mlp.experts.");
}

// Test: empty config_groups throws in DeepSeekV3 path
TEST(QuantConfigParser, DeepSeekV3Path_CompressedTensorsEmptyConfigGroupsThrows) {
  nlohmann::json config_json;
  config_json["quantization_config"] = {
      {"quant_method", "compressed-tensors"},
      {"config_groups", nlohmann::json::object()}};

  DeepSeekV3ConfigParser parser;
  auto ds_config = std::make_shared<DeepSeekV3Config>();
  ds_config->is_moe = true;
  ds_config->model_dir = "/tmp/nonexistent_model_dir";
  std::string yaml_quant = "auto";
  std::string yaml_gptq = "cutlass";

  EXPECT_ANY_THROW(parser.ParseQuantConfig(config_json, ds_config, yaml_quant, yaml_gptq));
}

// Test: standard GPTQ still works (regression check)
TEST(QuantConfigParser, DeepSeekV3Path_GptqStillWorks) {
  nlohmann::json config_json;
  config_json["quantization_config"] = {
      {"quant_method", "gptq"}, {"bits", 4}, {"group_size", 128}, {"desc_act", false}};

  DeepSeekV3ConfigParser parser;
  auto ds_config = std::make_shared<DeepSeekV3Config>();
  ds_config->is_moe = true;
  ds_config->model_dir = "/tmp/nonexistent_model_dir";
  std::string yaml_quant = "auto";
  std::string yaml_gptq = "cutlass";

  Status status = parser.ParseQuantConfig(config_json, ds_config, yaml_quant, yaml_gptq);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  EXPECT_FALSE(ds_config->is_compressed_tensors);
  EXPECT_EQ(ds_config->quant_config.method, QUANT_GPTQ);
  EXPECT_EQ(ds_config->quant_config.bits, 4u);
  EXPECT_EQ(ds_config->quant_config.group_size, 128u);
  EXPECT_FALSE(ds_config->quant_config.is_compressed_tensors);
}

}  // namespace ksana_llm
