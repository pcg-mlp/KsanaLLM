/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "configure/model_config.h"
#include "configure/model_config_parser.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

// ============================================================================
// Helpers
// ============================================================================

// Write a JSON object to a temporary file and return the file path.
static std::string WriteTempJson(const std::string &dir, const std::string &filename, const nlohmann::json &content) {
  std::filesystem::create_directories(dir);
  std::string path = dir + "/" + filename;
  std::ofstream ofs(path);
  ofs << content.dump(2);
  return path;
}

// ============================================================================
// GetModelDataType tests
// (tested indirectly via ParseModelConfig; here we exercise the logic through
//  a minimal config.json placed in a temp directory)
// ============================================================================

// Helper: build a minimal llama config.json JSON object.
static nlohmann::json BuildMinimalLlamaConfig(const std::string &dtype = "float16") {
  return {{"model_type", "llama"},       {"architectures", {"LlamaForCausalLM"}},
          {"torch_dtype", dtype},        {"hidden_size", 128},
          {"intermediate_size", 256},    {"max_position_embeddings", 1024},
          {"num_attention_heads", 2},    {"num_hidden_layers", 2},
          {"num_key_value_heads", 2},    {"rms_norm_eps", 1e-6},
          {"bos_token_id", 1},           {"eos_token_id", 2},
          {"pad_token_id", 0},           {"vocab_size", 300},
          {"tie_word_embeddings", false}};
}

// ============================================================================
// ParseModelMaxLength tests
// (exercised through ParseModelConfig with a real temp directory)
// ============================================================================

class ModelConfigParserTest : public ::testing::Test {
 protected:
  // Unique temp directory per test to avoid cross-test interference.
  std::string tmp_dir_;

  void SetUp() override {
    tmp_dir_ = std::string("/tmp/ksana_model_config_parser_test_") +
               std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
               ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::create_directories(tmp_dir_);
  }

  void TearDown() override { std::filesystem::remove_all(tmp_dir_); }

  // Write config.json into tmp_dir_ and run ParseModelConfig.
  Status RunParse(const nlohmann::json &config_json, ModelConfig &model_config,
                  const std::string &quant_method = "auto", const std::string &gptq_backend = "cutlass") {
    WriteTempJson(tmp_dir_, "config.json", config_json);
    EnvModelConfigParser parser(quant_method, gptq_backend);
    return parser.ParseModelConfig(tmp_dir_, tmp_dir_, "config.json", model_config);
  }
};

// ---------------------------------------------------------------------------
// Unsupported dtype must fail
// ---------------------------------------------------------------------------
TEST_F(ModelConfigParserTest, ParseUnsupportedDtypeThrows) {
  nlohmann::json cfg = BuildMinimalLlamaConfig("float32");
  ModelConfig model_config;
  EXPECT_ANY_THROW(RunParse(cfg, model_config));
}

// ---------------------------------------------------------------------------
// Missing all length keys must fail
// ---------------------------------------------------------------------------
TEST_F(ModelConfigParserTest, ParseMissingMaxLengthThrows) {
  nlohmann::json cfg = BuildMinimalLlamaConfig();
  cfg.erase("max_position_embeddings");
  ModelConfig model_config;
  EXPECT_ANY_THROW(RunParse(cfg, model_config));
}

// ---------------------------------------------------------------------------
// rope_scaling with linear factor scales max_training_seq_len
// ---------------------------------------------------------------------------
TEST_F(ModelConfigParserTest, ParseRopeScalingLinearFactor) {
  nlohmann::json cfg = BuildMinimalLlamaConfig();
  cfg["max_position_embeddings"] = 1024;
  cfg["rope_scaling"] = {{"type", "linear"}, {"factor", 2.0f}};
  ModelConfig model_config;
  Status status = RunParse(cfg, model_config);

  EXPECT_TRUE(status.OK()) << status.GetMessage();
  // linear type is not in the "no-scale" set, so derived_max_model_len *= factor
  EXPECT_EQ(model_config.max_training_seq_len, 2048u);
  EXPECT_FLOAT_EQ(model_config.rope_scaling_factor_config.factor, 2.0f);
}

// ---------------------------------------------------------------------------
// rope_scaling with yarn type uses original_max_position_embeddings * factor
// ---------------------------------------------------------------------------
TEST_F(ModelConfigParserTest, ParseRopeScalingYarn) {
  nlohmann::json cfg = BuildMinimalLlamaConfig();
  cfg["max_position_embeddings"] = 131072;
  cfg["rope_scaling"] = {{"type", "yarn"}, {"factor", 4.0f}, {"original_max_position_embeddings", 32768}};
  ModelConfig model_config;
  Status status = RunParse(cfg, model_config);

  EXPECT_TRUE(status.OK()) << status.GetMessage();
  EXPECT_EQ(model_config.max_training_seq_len, 131072u);
  EXPECT_EQ(model_config.rope_scaling_factor_config.type, "yarn");
  EXPECT_EQ(model_config.rope_scaling_factor_config.original_max_position_embeddings, 32768);
}

// ---------------------------------------------------------------------------
// rope_scaling with llama3 type stores low/high freq factors
// ---------------------------------------------------------------------------
TEST_F(ModelConfigParserTest, ParseRopeScalingLlama3) {
  nlohmann::json cfg = BuildMinimalLlamaConfig();
  cfg["max_position_embeddings"] = 8192;
  cfg["rope_scaling"] = {{"type", "llama3"},
                         {"factor", 8.0f},
                         {"low_freq_factor", 1.0f},
                         {"high_freq_factor", 4.0f},
                         {"original_max_position_embeddings", 8192}};
  ModelConfig model_config;
  Status status = RunParse(cfg, model_config);

  EXPECT_TRUE(status.OK()) << status.GetMessage();
  EXPECT_EQ(model_config.rope_scaling_factor_config.type, "llama3");
  EXPECT_FLOAT_EQ(model_config.rope_scaling_factor_config.low_freq_factor, 1.0f);
  EXPECT_FLOAT_EQ(model_config.rope_scaling_factor_config.high_freq_factor, 4.0f);
  EXPECT_EQ(model_config.rope_scaling_factor_config.original_max_position_embeddings, 8192);
}

// ---------------------------------------------------------------------------
// generation_config.json overrides eos_token_id (single value)
// ---------------------------------------------------------------------------
TEST_F(ModelConfigParserTest, GenerationConfigOverridesEndId) {
  nlohmann::json cfg = BuildMinimalLlamaConfig();
  ModelConfig model_config;

  // Write generation_config.json with a different eos_token_id.
  WriteTempJson(tmp_dir_, "generation_config.json", {{"eos_token_id", 99}});

  Status status = RunParse(cfg, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();
  ASSERT_EQ(model_config.end_ids.size(), 1u);
  EXPECT_EQ(model_config.end_ids[0], 99u);
}

// ---------------------------------------------------------------------------
// generation_config.json overrides eos_token_id (array value)
// ---------------------------------------------------------------------------
TEST_F(ModelConfigParserTest, GenerationConfigOverridesEndIdArray) {
  nlohmann::json cfg = BuildMinimalLlamaConfig();
  ModelConfig model_config;

  WriteTempJson(tmp_dir_, "generation_config.json", {{"eos_token_id", {10, 20, 30}}});

  Status status = RunParse(cfg, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();
  ASSERT_EQ(model_config.end_ids.size(), 3u);
  EXPECT_EQ(model_config.end_ids[0], 10u);
  EXPECT_EQ(model_config.end_ids[1], 20u);
  EXPECT_EQ(model_config.end_ids[2], 30u);
}

// ---------------------------------------------------------------------------
// generation_config.json without eos_token_id leaves end_ids unchanged
// ---------------------------------------------------------------------------
TEST_F(ModelConfigParserTest, GenerationConfigNoEosTokenIdKeepsOriginal) {
  nlohmann::json cfg = BuildMinimalLlamaConfig();
  ModelConfig model_config;

  // generation_config.json exists but has no eos_token_id field.
  WriteTempJson(tmp_dir_, "generation_config.json", {{"temperature", 0.7}});

  Status status = RunParse(cfg, model_config);
  EXPECT_TRUE(status.OK()) << status.GetMessage();
  ASSERT_EQ(model_config.end_ids.size(), 1u);
  EXPECT_EQ(model_config.end_ids[0], 2u);  // original from config.json
}

// ---------------------------------------------------------------------------
// Helper: build a Qwen3-omni-style nested config with thinker_config/text_config.
// The outer model_type triggers the moeconformerllm/qwen3_omni_moe branch;
// inner text_config holds the real MoE parameters.
// ---------------------------------------------------------------------------
static nlohmann::json BuildQwen3OmniNestedConfig(const std::string &outer_model_type) {
  nlohmann::json text_config = {{"model_type", "qwen3_moe"},
                                {"architectures", {"Qwen3MoeForCausalLM"}},
                                {"torch_dtype", "bfloat16"},
                                {"hidden_size", 128},
                                {"intermediate_size", 256},
                                {"moe_intermediate_size", 64},
                                {"max_position_embeddings", 4096},
                                {"num_attention_heads", 4},
                                {"num_hidden_layers", 2},
                                {"num_key_value_heads", 2},
                                {"num_experts", 8},
                                {"num_experts_per_tok", 2},
                                {"norm_topk_prob", true},
                                {"rms_norm_eps", 1e-6},
                                {"rope_theta", 1000000.0f},
                                {"rope_scaling", {{"type", "mrope"}, {"mrope_section", {16, 24, 24}}}},
                                {"bos_token_id", 151643},
                                {"eos_token_id", 151645},
                                {"pad_token_id", 151643},
                                {"vocab_size", 152064},
                                {"tie_word_embeddings", false}};
  nlohmann::json thinker_config = {{"text_config", text_config}};
  return {{"model_type", outer_model_type},
          {"architectures", {"Qwen3OmniMoeForConditionalGeneration"}},
          {"torch_dtype", "bfloat16"},
          {"thinker_config", thinker_config}};
}

// ---------------------------------------------------------------------------
// moeconformerllm and qwen3_omni_moe share one branch: nested thinker_config/
// text_config is unwrapped, type rewritten to qwen3_moe, MoE attributes
// populated, and rope_scaling.mrope_section stripped so type is not promoted
// to "mrope" (keeps the fast default 1-D rope path for text-only inference).
// ---------------------------------------------------------------------------
TEST_F(ModelConfigParserTest, ParseQwen3OmniNestedConfig) {
  for (const std::string &outer_type : {"moeconformerllm", "qwen3_omni_moe"}) {
    SCOPED_TRACE("outer model_type=" + outer_type);
    // Fresh temp dir per iteration so each RunParse gets a clean config.json.
    std::filesystem::remove_all(tmp_dir_);
    std::filesystem::create_directories(tmp_dir_);

    nlohmann::json cfg = BuildQwen3OmniNestedConfig(outer_type);
    ModelConfig model_config;
    Status status = RunParse(cfg, model_config);

    EXPECT_TRUE(status.OK()) << status.GetMessage();
    EXPECT_EQ(model_config.type, "qwen3_moe");
    EXPECT_TRUE(model_config.is_moe);
    EXPECT_EQ(model_config.moe_config.num_experts, 8u);
    EXPECT_EQ(model_config.moe_config.experts_topk, 2u);
    EXPECT_EQ(model_config.moe_config.moe_inter_size, 64u);
    EXPECT_TRUE(model_config.moe_config.norm_topk_prob);
    // Inner text_config params reach ModelConfig (proves nested extraction).
    EXPECT_EQ(model_config.num_layer, 2u);
    EXPECT_EQ(model_config.hidden_units, 128u);
    EXPECT_EQ(model_config.vocab_size, 152064u);
    // mrope_section stripped, type coerced to "default" → not promoted to "mrope".
    EXPECT_NE(model_config.rope_scaling_factor_config.type, "mrope");
    EXPECT_TRUE(model_config.rope_scaling_factor_config.mrope_section.empty());
  }
}

// ---------------------------------------------------------------------------
// Missing config.json returns an error status
// ---------------------------------------------------------------------------
TEST_F(ModelConfigParserTest, MissingConfigJsonReturnsError) {
  // Do NOT write config.json; the directory exists but is empty.
  ModelConfig model_config;
  EnvModelConfigParser parser("auto", "cutlass");
  Status status = parser.ParseModelConfig(tmp_dir_, tmp_dir_, "config.json", model_config);
  EXPECT_FALSE(status.OK());
}

// ============================================================================
// ApplyJsonModelOverride tests
// ============================================================================

class ApplyJsonModelOverrideTest : public ::testing::Test {
 protected:
  ModelConfig model_config_;

  void SetUp() override {
    // Populate a baseline ModelConfig so we can verify selective overrides.
    model_config_.num_layer = 32;
    model_config_.hidden_units = 4096;
    model_config_.head_num = 32;
    model_config_.num_key_value_heads = 8;
    model_config_.inter_size = 11008;
    model_config_.size_per_head = 128;
    model_config_.vocab_size = 32000;
    model_config_.rotary_embedding = 128;
    model_config_.max_training_seq_len = 4096;
    model_config_.max_position_embeddings = 4096;
    model_config_.rope_theta = 10000.0f;
    model_config_.layernorm_eps = 1e-5f;
    model_config_.name = "original_name";
    model_config_.type = "llama";
  }
};

// ---------------------------------------------------------------------------
// Empty override JSON leaves all fields unchanged
// ---------------------------------------------------------------------------
TEST_F(ApplyJsonModelOverrideTest, EmptyOverrideChangesNothing) {
  nlohmann::json override_json = nlohmann::json::object();
  ApplyJsonModelOverride(override_json, model_config_);

  EXPECT_EQ(model_config_.num_layer, 32u);
  EXPECT_EQ(model_config_.hidden_units, 4096u);
  EXPECT_EQ(model_config_.head_num, 32u);
  EXPECT_EQ(model_config_.num_key_value_heads, 8u);
  EXPECT_EQ(model_config_.inter_size, 11008u);
  EXPECT_EQ(model_config_.size_per_head, 128u);
  EXPECT_EQ(model_config_.vocab_size, 32000u);
  EXPECT_EQ(model_config_.rotary_embedding, 128u);
  EXPECT_EQ(model_config_.max_training_seq_len, 4096u);
  EXPECT_EQ(model_config_.max_position_embeddings, 4096);
  EXPECT_FLOAT_EQ(model_config_.rope_theta, 10000.0f);
  EXPECT_FLOAT_EQ(model_config_.layernorm_eps, 1e-5f);
  EXPECT_EQ(model_config_.name, "original_name");
  EXPECT_EQ(model_config_.type, "llama");
}

// ---------------------------------------------------------------------------
// Override a single integer field
// ---------------------------------------------------------------------------
TEST_F(ApplyJsonModelOverrideTest, OverrideSingleIntField) {
  nlohmann::json override_json = {{"num_layer", 64}};
  ApplyJsonModelOverride(override_json, model_config_);

  EXPECT_EQ(model_config_.num_layer, 64u);
  // All other fields must remain unchanged.
  EXPECT_EQ(model_config_.hidden_units, 4096u);
  EXPECT_EQ(model_config_.head_num, 32u);
}

// ---------------------------------------------------------------------------
// Override multiple fields at once
// ---------------------------------------------------------------------------
TEST_F(ApplyJsonModelOverrideTest, OverrideMultipleFields) {
  nlohmann::json override_json = {{"hidden_units", 2048},
                                  {"head_num", 16},
                                  {"num_key_value_heads", 4},
                                  {"inter_size", 5504},
                                  {"size_per_head", 128},
                                  {"vocab_size", 50000},
                                  {"rotary_embedding", 64},
                                  {"max_training_seq_len", 8192},
                                  {"max_position_embeddings", 8192},
                                  {"rope_theta", 500000.0f},
                                  {"layernorm_eps", 1e-6f},
                                  {"name", "new_name"},
                                  {"type", "qwen2"}};
  ApplyJsonModelOverride(override_json, model_config_);

  EXPECT_EQ(model_config_.hidden_units, 2048u);
  EXPECT_EQ(model_config_.head_num, 16u);
  EXPECT_EQ(model_config_.num_key_value_heads, 4u);
  EXPECT_EQ(model_config_.inter_size, 5504u);
  EXPECT_EQ(model_config_.size_per_head, 128u);
  EXPECT_EQ(model_config_.vocab_size, 50000u);
  EXPECT_EQ(model_config_.rotary_embedding, 64u);
  EXPECT_EQ(model_config_.max_training_seq_len, 8192u);
  EXPECT_EQ(model_config_.max_position_embeddings, 8192);
  EXPECT_FLOAT_EQ(model_config_.rope_theta, 500000.0f);
  EXPECT_FLOAT_EQ(model_config_.layernorm_eps, 1e-6f);
  EXPECT_EQ(model_config_.name, "new_name");
  EXPECT_EQ(model_config_.type, "qwen2");
}

// ---------------------------------------------------------------------------
// Override only float fields
// ---------------------------------------------------------------------------
TEST_F(ApplyJsonModelOverrideTest, OverrideFloatFields) {
  nlohmann::json override_json = {{"rope_theta", 1000000.0f}, {"layernorm_eps", 1e-8f}};
  ApplyJsonModelOverride(override_json, model_config_);

  EXPECT_FLOAT_EQ(model_config_.rope_theta, 1000000.0f);
  EXPECT_FLOAT_EQ(model_config_.layernorm_eps, 1e-8f);
  // Integer fields unchanged.
  EXPECT_EQ(model_config_.num_layer, 32u);
}

// ---------------------------------------------------------------------------
// Override only string fields
// ---------------------------------------------------------------------------
TEST_F(ApplyJsonModelOverrideTest, OverrideStringFields) {
  nlohmann::json override_json = {{"name", "my_model"}, {"type", "deepseek_v3"}};
  ApplyJsonModelOverride(override_json, model_config_);

  EXPECT_EQ(model_config_.name, "my_model");
  EXPECT_EQ(model_config_.type, "deepseek_v3");
  // Numeric fields unchanged.
  EXPECT_EQ(model_config_.num_layer, 32u);
}

// ---------------------------------------------------------------------------
// Unknown fields in override JSON are silently ignored
// ---------------------------------------------------------------------------
TEST_F(ApplyJsonModelOverrideTest, UnknownFieldsAreIgnored) {
  nlohmann::json override_json = {{"totally_unknown_field", 42}, {"another_unknown", "hello"}};
  // Should not throw and should not modify any known field.
  EXPECT_NO_THROW(ApplyJsonModelOverride(override_json, model_config_));
  EXPECT_EQ(model_config_.num_layer, 32u);
  EXPECT_EQ(model_config_.type, "llama");
}

// ---------------------------------------------------------------------------
// Type-mismatched override emits a warning and skips the field (no throw)
// ---------------------------------------------------------------------------
TEST_F(ApplyJsonModelOverrideTest, TypeMismatchDoesNotThrow) {
  // Provide a string where an integer is expected.
  nlohmann::json override_json = {{"num_layer", "not_an_integer"}};
  EXPECT_NO_THROW(ApplyJsonModelOverride(override_json, model_config_));
  // The field must remain at its original value because the override was skipped.
  EXPECT_EQ(model_config_.num_layer, 32u);
}

// ---------------------------------------------------------------------------
// Override num_layer to zero (boundary value)
// ---------------------------------------------------------------------------
TEST_F(ApplyJsonModelOverrideTest, OverrideNumLayerToZero) {
  nlohmann::json override_json = {{"num_layer", 0}};
  ApplyJsonModelOverride(override_json, model_config_);
  EXPECT_EQ(model_config_.num_layer, 0u);
}

// ---------------------------------------------------------------------------
// Override max_position_embeddings (signed int field)
// ---------------------------------------------------------------------------
TEST_F(ApplyJsonModelOverrideTest, OverrideMaxPositionEmbeddings) {
  nlohmann::json override_json = {{"max_position_embeddings", 131072}};
  ApplyJsonModelOverride(override_json, model_config_);
  EXPECT_EQ(model_config_.max_position_embeddings, 131072);
}

// ---------------------------------------------------------------------------
// Partial override: only some fields present in JSON
// ---------------------------------------------------------------------------
TEST_F(ApplyJsonModelOverrideTest, PartialOverrideOnlyAffectsSpecifiedFields) {
  nlohmann::json override_json = {{"vocab_size", 128256}};
  ApplyJsonModelOverride(override_json, model_config_);

  EXPECT_EQ(model_config_.vocab_size, 128256u);
  // All other fields must be untouched.
  EXPECT_EQ(model_config_.num_layer, 32u);
  EXPECT_EQ(model_config_.hidden_units, 4096u);
  EXPECT_EQ(model_config_.head_num, 32u);
  EXPECT_EQ(model_config_.inter_size, 11008u);
  EXPECT_EQ(model_config_.name, "original_name");
  EXPECT_EQ(model_config_.type, "llama");
}

}  // namespace ksana_llm
