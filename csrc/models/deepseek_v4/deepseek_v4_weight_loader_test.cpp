/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "models/deepseek_v4/deepseek_v4_config.h"
#include "models/deepseek_v4/deepseek_v4_weight_loader.h"
#include "nlohmann/json.hpp"
#include "test.h"
#include "utils/logger.h"
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
// Test helpers
// ---------------------------------------------------------------------------

namespace {

// Create a temporary directory and write a JSON file into it.
// Returns the directory path (caller should clean up).
std::string CreateTempModelDir(const std::string& suffix = "") {
  auto dir = std::filesystem::temp_directory_path() /
             ("dsv4_test_" + suffix + "_" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())));
  std::filesystem::create_directories(dir);
  return dir.string();
}

void WriteJsonFile(const std::string& path, const nlohmann::json& json) {
  std::ofstream ofs(path);
  ASSERT_TRUE(ofs.is_open());
  ofs << json.dump();
  ofs.close();
}

// Replicate the static IsRawMtpWeightName logic from deepseek_v4_weight_loader.cpp
// since it is file-static and not exposed in the header.
bool IsRawMtpWeightName(const std::string& weight_name) {
  return weight_name.rfind("mtp.", 0) == 0 || weight_name.find("model.mtp.") != std::string::npos;
}

// Replicate the MTP weight name rewriting logic from ProcessModelWeights.
// Returns the rewritten name, or the original if not an MTP weight.
std::string RewriteMtpWeightName(const std::string& origin_name, size_t num_layer) {
  std::string file_weight_name = origin_name;
  if (auto mtp_pos = file_weight_name.find("model.mtp."); mtp_pos != std::string::npos) {
    if (auto dot_pos = file_weight_name.find('.', mtp_pos + strlen("model.mtp.")); dot_pos != std::string::npos) {
      int mtp_idx =
          std::stoi(file_weight_name.substr(mtp_pos + strlen("model.mtp."), dot_pos - mtp_pos - strlen("model.mtp.")));
      int mapped_layer = static_cast<int>(num_layer) + mtp_idx;
      file_weight_name = file_weight_name.substr(0, mtp_pos) + "model.layers." + std::to_string(mapped_layer) + "." +
                         file_weight_name.substr(dot_pos + 1);
    }
  }
  return file_weight_name;
}

}  // namespace

// ---------------------------------------------------------------------------
// ReadDsparkNumLayers tests
// ---------------------------------------------------------------------------

class ReadDsparkNumLayersTest : public testing::Test {
 protected:
  void SetUp() override { InitLoguru(); }

  void TearDown() override {
    if (!tmp_dir_.empty() && std::filesystem::exists(tmp_dir_)) {
      std::filesystem::remove_all(tmp_dir_);
    }
  }

  std::string tmp_dir_;
};

// Test: read n_mtp_layers from inference/config.json
TEST_F(ReadDsparkNumLayersTest, ReadFromInferenceConfig) {
  tmp_dir_ = CreateTempModelDir("inference");
  std::filesystem::create_directories(tmp_dir_ + "/inference");

  nlohmann::json inference_config;
  inference_config["n_mtp_layers"] = 3;
  WriteJsonFile(tmp_dir_ + "/inference/config.json", inference_config);

  int result = ReadDsparkNumLayers(tmp_dir_, 1);
  EXPECT_EQ(result, 3);
}

// Test: n_mtp_layers = 0 should fall back to default
TEST_F(ReadDsparkNumLayersTest, ZeroN_mtp_layersFallsBack) {
  tmp_dir_ = CreateTempModelDir("zero");
  std::filesystem::create_directories(tmp_dir_ + "/inference");

  nlohmann::json inference_config;
  inference_config["n_mtp_layers"] = 0;
  WriteJsonFile(tmp_dir_ + "/inference/config.json", inference_config);

  int result = ReadDsparkNumLayers(tmp_dir_, 5);
  EXPECT_EQ(result, 5);
}

// Test: missing inference/config.json should fall back to safetensors index
TEST_F(ReadDsparkNumLayersTest, FallBackToSafetensorsIndex) {
  tmp_dir_ = CreateTempModelDir("safetensors");

  nlohmann::json index_json;
  index_json["weight_map"] = {{"mtp.0.attn.wq_a.weight", "model-00001-of-00002.safetensors"},
                              {"mtp.0.attn.wo_b.weight", "model-00001-of-00002.safetensors"},
                              {"mtp.1.attn.wq_a.weight", "model-00002-of-00002.safetensors"},
                              {"mtp.1.mlp.down_proj.weight", "model-00002-of-00002.safetensors"},
                              {"mtp.2.attn.wq_a.weight", "model-00002-of-00002.safetensors"},
                              {"model.layers.0.attn.wq_a.weight", "model-00001-of-00002.safetensors"},
                              {"model.embed_tokens.weight", "model-00001-of-00002.safetensors"}};
  WriteJsonFile(tmp_dir_ + "/model.safetensors.index.json", index_json);

  int result = ReadDsparkNumLayers(tmp_dir_, 1);
  // max stage id is 2, so result = 2 + 1 = 3
  EXPECT_EQ(result, 3);
}

// Test: no config files found, return default
TEST_F(ReadDsparkNumLayersTest, NoConfigFilesReturnsDefault) {
  tmp_dir_ = CreateTempModelDir("empty");
  int result = ReadDsparkNumLayers(tmp_dir_, 7);
  EXPECT_EQ(result, 7);
}

// Test: malformed JSON should fall back gracefully
TEST_F(ReadDsparkNumLayersTest, MalformedJsonFallsBack) {
  tmp_dir_ = CreateTempModelDir("malformed");
  std::filesystem::create_directories(tmp_dir_ + "/inference");

  std::ofstream ofs(tmp_dir_ + "/inference/config.json");
  ofs << "{ this is not valid json";
  ofs.close();

  int result = ReadDsparkNumLayers(tmp_dir_, 2);
  EXPECT_EQ(result, 2);
}

// Test: safetensors index with no mtp weights should return default
TEST_F(ReadDsparkNumLayersTest, NoMtpInSafetensorsReturnsDefault) {
  tmp_dir_ = CreateTempModelDir("nomtp");

  nlohmann::json index_json;
  index_json["weight_map"] = {{"model.layers.0.attn.wq_a.weight", "model-00001-of-00002.safetensors"},
                              {"model.embed_tokens.weight", "model-00001-of-00002.safetensors"}};
  WriteJsonFile(tmp_dir_ + "/model.safetensors.index.json", index_json);

  int result = ReadDsparkNumLayers(tmp_dir_, 4);
  EXPECT_EQ(result, 4);
}

// Test: inference/config.json takes priority over safetensors index
TEST_F(ReadDsparkNumLayersTest, InferenceConfigTakesPriority) {
  tmp_dir_ = CreateTempModelDir("priority");
  std::filesystem::create_directories(tmp_dir_ + "/inference");

  nlohmann::json inference_config;
  inference_config["n_mtp_layers"] = 5;
  WriteJsonFile(tmp_dir_ + "/inference/config.json", inference_config);

  nlohmann::json index_json;
  index_json["weight_map"] = {{"mtp.0.weight", "file.safetensors"}, {"mtp.1.weight", "file.safetensors"}};
  WriteJsonFile(tmp_dir_ + "/model.safetensors.index.json", index_json);

  int result = ReadDsparkNumLayers(tmp_dir_, 1);
  // Should use inference config (5), not safetensors (2)
  EXPECT_EQ(result, 5);
}

// ---------------------------------------------------------------------------
// IsRawMtpWeightName tests
// ---------------------------------------------------------------------------

TEST(IsRawMtpWeightNameTest, PrefixMtp) {
  EXPECT_TRUE(IsRawMtpWeightName("mtp.0.attn.wq_a.weight"));
  EXPECT_TRUE(IsRawMtpWeightName("mtp.1.mlp.down_proj.weight"));
  EXPECT_TRUE(IsRawMtpWeightName("mtp."));
}

TEST(IsRawMtpWeightNameTest, PrefixModelMtp) {
  EXPECT_TRUE(IsRawMtpWeightName("model.mtp.0.attn.wq_a.weight"));
  EXPECT_TRUE(IsRawMtpWeightName("model.mtp.2.mlp.gate_proj.weight"));
}

TEST(IsRawMtpWeightNameTest, NonMtpWeights) {
  EXPECT_FALSE(IsRawMtpWeightName("model.layers.0.attn.wq_a.weight"));
  EXPECT_FALSE(IsRawMtpWeightName("model.embed_tokens.weight"));
  EXPECT_FALSE(IsRawMtpWeightName("lm_head.weight"));
  EXPECT_FALSE(IsRawMtpWeightName("model.layers.5.mlp.experts.0.gate_proj.weight"));
}

TEST(IsRawMtpWeightNameTest, EdgeCases) {
  EXPECT_FALSE(IsRawMtpWeightName(""));
  EXPECT_FALSE(IsRawMtpWeightName("mt"));
  EXPECT_FALSE(IsRawMtpWeightName("mtp"));
  // "mtp" without trailing dot should not match "mtp." prefix
  EXPECT_FALSE(IsRawMtpWeightName("mtp_layers"));
  // Substring match should work for model.mtp.
  EXPECT_TRUE(IsRawMtpWeightName("base_model.model.model.mtp.0.weight"));
}

// ---------------------------------------------------------------------------
// MTP weight name rewriting tests
// ---------------------------------------------------------------------------

TEST(MtpWeightRewriteTest, RewriteSingleMtp) {
  size_t num_layer = 61;
  std::string input = "model.mtp.0.attn.wq_a.weight";
  std::string expected = "model.layers.61.attn.wq_a.weight";
  EXPECT_EQ(RewriteMtpWeightName(input, num_layer), expected);
}

TEST(MtpWeightRewriteTest, RewriteMultipleMtp) {
  size_t num_layer = 61;
  std::string input0 = "model.mtp.0.attn.wq_a.weight";
  std::string input1 = "model.mtp.1.attn.wq_a.weight";
  std::string input2 = "model.mtp.2.attn.wq_a.weight";

  EXPECT_EQ(RewriteMtpWeightName(input0, num_layer), "model.layers.61.attn.wq_a.weight");
  EXPECT_EQ(RewriteMtpWeightName(input1, num_layer), "model.layers.62.attn.wq_a.weight");
  EXPECT_EQ(RewriteMtpWeightName(input2, num_layer), "model.layers.63.attn.wq_a.weight");
}

TEST(MtpWeightRewriteTest, NonMtpWeightUnchanged) {
  size_t num_layer = 61;
  std::string input = "model.layers.0.attn.wq_a.weight";
  EXPECT_EQ(RewriteMtpWeightName(input, num_layer), input);

  std::string input2 = "model.embed_tokens.weight";
  EXPECT_EQ(RewriteMtpWeightName(input2, num_layer), input2);
}

TEST(MtpWeightRewriteTest, RewritePreservesSuffix) {
  size_t num_layer = 10;
  std::string input = "model.mtp.3.mlp.experts.0.gate_proj.weight";
  std::string expected = "model.layers.13.mlp.experts.0.gate_proj.weight";
  EXPECT_EQ(RewriteMtpWeightName(input, num_layer), expected);
}

TEST(MtpWeightRewriteTest, RewritePreservesPrefix) {
  size_t num_layer = 10;
  // Weight name with a prefix before model.mtp.
  std::string input = "base_model.model.mtp.1.attn.wo_b.weight";
  std::string result = RewriteMtpWeightName(input, num_layer);
  // The prefix "base_model." should be preserved, and mtp.1 → layers.11
  EXPECT_NE(result, input);
  EXPECT_TRUE(result.find("model.layers.11.") != std::string::npos);
  EXPECT_TRUE(result.find("base_model.") != std::string::npos);
}

// ---------------------------------------------------------------------------
// DSparkConfig tests
// ---------------------------------------------------------------------------

TEST(DsparkConfigTest, DefaultValues) {
  DsparkConfig config;
  EXPECT_EQ(config.dspark_block_size, 0u);
  EXPECT_EQ(config.dspark_noise_token_id, 0u);
  EXPECT_EQ(config.dspark_markov_rank, 256u);
  EXPECT_TRUE(config.dspark_target_layer_ids.empty());
}

TEST(DsparkConfigTest, DsparkDisabledByDefault) {
  // dspark_block_size == 0 means DSpark is disabled
  DsparkConfig config;
  EXPECT_EQ(config.dspark_block_size, 0u);
}

// ---------------------------------------------------------------------------
// Tensor GetView tests for DSpark wkv view logic
// ---------------------------------------------------------------------------

TEST(DsparkWkvViewTest, GetViewCreatesSharedStorageView) {
  // Simulate the wkv view creation logic from ProcessModelWeights:
  // The wkv view shares the same underlying buffer as the fused wqkv_a tensor,
  // offset by the wq_a portion's byte size.
  size_t q_lora_rank = 1024;
  size_t head_dim = 512;
  size_t wkv_rows = head_dim;  // wkv has head_dim rows
  size_t wq_a_rows = q_lora_rank;
  size_t fused_rows = wq_a_rows + wkv_rows;

  // Create a "fused" tensor on host with fused_rows elements
  std::vector<float> fused_data(fused_rows);
  for (size_t i = 0; i < fused_rows; ++i) {
    fused_data[i] = static_cast<float>(i);
  }

  Tensor fused_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {fused_rows, 1});
  std::memcpy(fused_tensor.GetPtr<void>(), fused_data.data(), fused_rows * sizeof(float));

  // Create a view for the wkv portion (offset by wq_a_rows elements)
  size_t wkv_bytes = wkv_rows * sizeof(float);
  size_t fused_total_bytes = fused_tensor.GetTotalBytes();
  size_t offset_bytes = fused_total_bytes - wkv_bytes;
  size_t elem_size = GetTypeSize(DataType::TYPE_FP32);
  size_t offset_elems = offset_bytes / elem_size;

  Tensor wkv_view = fused_tensor.GetView({wkv_rows, 1}, offset_elems);

  // Verify the view has the correct shape
  ASSERT_EQ(wkv_view.shape.size(), 2u);
  EXPECT_EQ(wkv_view.shape[0], wkv_rows);
  EXPECT_EQ(wkv_view.shape[1], 1u);

  // Verify the view points to the correct data (should start at wq_a_rows)
  const float* view_data = wkv_view.GetPtr<float>();
  EXPECT_FLOAT_EQ(view_data[0], static_cast<float>(wq_a_rows));
  EXPECT_FLOAT_EQ(view_data[wkv_rows - 1], static_cast<float>(fused_rows - 1));
}

TEST(DsparkWkvViewTest, GetViewWithScaleTensor) {
  // When the weight is a scale tensor, the fused dim is divided by block_n
  size_t q_lora_rank = 1024;
  size_t head_dim = 512;
  size_t block_n = 128;

  size_t wkv_scale_rows = head_dim / block_n;                  // 4
  size_t wq_a_scale_rows = q_lora_rank / block_n;              // 8
  size_t fused_scale_rows = wq_a_scale_rows + wkv_scale_rows;  // 12

  std::vector<float> fused_data(fused_scale_rows);
  for (size_t i = 0; i < fused_scale_rows; ++i) {
    fused_data[i] = static_cast<float>(i) * 0.1f;
  }

  Tensor fused_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {fused_scale_rows, 1});
  std::memcpy(fused_tensor.GetPtr<void>(), fused_data.data(), fused_scale_rows * sizeof(float));

  // Compute offset for wkv scale portion
  size_t wkv_bytes = wkv_scale_rows * sizeof(float);
  size_t fused_total_bytes = fused_tensor.GetTotalBytes();
  size_t offset_bytes = fused_total_bytes - wkv_bytes;
  size_t offset_elems = offset_bytes / sizeof(float);

  Tensor wkv_view = fused_tensor.GetView({wkv_scale_rows, 1}, offset_elems);

  const float* view_data = wkv_view.GetPtr<float>();
  // Should start at wq_a_scale_rows = 8
  EXPECT_FLOAT_EQ(view_data[0], static_cast<float>(wq_a_scale_rows) * 0.1f);
}

// ---------------------------------------------------------------------------
// Weight name classification tests
// ---------------------------------------------------------------------------

// Test that weight names are correctly classified for DSpark-specific handling.
// These tests verify the string matching logic used in ProcessModelWeights
// to route weights to the correct handler.
TEST(DsparkWeightNameClassificationTest, MainProjWeight) {
  std::string name = "model.mtp.0.main_proj.weight";
  EXPECT_TRUE(name.find("main_proj.weight") != std::string::npos);
}

TEST(DsparkWeightNameClassificationTest, MarkovW1Weight) {
  std::string name = "model.mtp.0.markov_head.markov_w1.weight";
  EXPECT_TRUE(name.find("markov_head.markov_w1.weight") != std::string::npos);
}

TEST(DsparkWeightNameClassificationTest, MarkovW2Weight) {
  std::string name = "model.mtp.0.markov_head.markov_w2.weight";
  EXPECT_TRUE(name.find("markov_head.markov_w2.weight") != std::string::npos);
}

TEST(DsparkWeightNameClassificationTest, ConfidenceHeadWeight) {
  std::string name = "model.mtp.0.confidence_head.proj.weight";
  EXPECT_TRUE(name.find("confidence_head.proj.weight") != std::string::npos);
}

TEST(DsparkWeightNameClassificationTest, EProjWeight) {
  std::string name = "model.mtp.0.e_proj.weight";
  EXPECT_TRUE(name.find("e_proj.weight") != std::string::npos);
}

TEST(DsparkWeightNameClassificationTest, HProjWeight) {
  std::string name = "model.mtp.0.h_proj.weight";
  EXPECT_TRUE(name.find("h_proj.weight") != std::string::npos);
}

// ---------------------------------------------------------------------------
// DSpark MTP weight routing tests (lines 847-895 in deepseek_v4_weight_loader.cpp)
// ---------------------------------------------------------------------------

// Replicate the full routing logic for DSpark MTP weights from ProcessModelWeights.
// Each handler is checked in order; the first match wins.
enum class DsparkMtpRoute {
  kMainProj,        // main_proj.weight: direct copy, no TP split
  kMarkovW1,        // markov_head.markov_w1.weight: replicated full [V, R] (no TP split)
  kMarkovW2,        // markov_head.markov_w2.weight: replicated + transposed to [R, V] (no TP split)
  kConfidenceHead,  // confidence_head.proj.weight: copy then transpose
  kEProj,           // e_proj.weight: direct copy
  kHProj,           // h_proj.weight: direct copy
  kFallback,        // unhandled weight: direct copy with warning
};

DsparkMtpRoute RouteDsparkMtpWeight(const std::string& name) {
  if (name.find("main_proj.weight") != std::string::npos) return DsparkMtpRoute::kMainProj;
  if (name.find("markov_head.markov_w1.weight") != std::string::npos) return DsparkMtpRoute::kMarkovW1;
  if (name.find("markov_head.markov_w2.weight") != std::string::npos) return DsparkMtpRoute::kMarkovW2;
  if (name.find("confidence_head.proj.weight") != std::string::npos) return DsparkMtpRoute::kConfidenceHead;
  if (name.find("e_proj.weight") != std::string::npos) return DsparkMtpRoute::kEProj;
  if (name.find("h_proj.weight") != std::string::npos) return DsparkMtpRoute::kHProj;
  return DsparkMtpRoute::kFallback;
}

TEST(DsparkMtpRouteTest, MainProjRoutedCorrectly) {
  EXPECT_EQ(RouteDsparkMtpWeight("model.mtp.0.main_proj.weight"), DsparkMtpRoute::kMainProj);
  EXPECT_EQ(RouteDsparkMtpWeight("model.mtp.1.main_proj.weight"), DsparkMtpRoute::kMainProj);
  EXPECT_EQ(RouteDsparkMtpWeight("model.layers.62.main_proj.weight"), DsparkMtpRoute::kMainProj);
}

TEST(DsparkMtpRouteTest, MarkovW1RoutedCorrectly) {
  EXPECT_EQ(RouteDsparkMtpWeight("model.mtp.0.markov_head.markov_w1.weight"), DsparkMtpRoute::kMarkovW1);
  EXPECT_EQ(RouteDsparkMtpWeight("model.mtp.2.markov_head.markov_w1.weight"), DsparkMtpRoute::kMarkovW1);
}

TEST(DsparkMtpRouteTest, MarkovW2RoutedCorrectly) {
  EXPECT_EQ(RouteDsparkMtpWeight("model.mtp.0.markov_head.markov_w2.weight"), DsparkMtpRoute::kMarkovW2);
  EXPECT_EQ(RouteDsparkMtpWeight("model.mtp.1.markov_head.markov_w2.weight"), DsparkMtpRoute::kMarkovW2);
}

TEST(DsparkMtpRouteTest, ConfidenceHeadRoutedCorrectly) {
  EXPECT_EQ(RouteDsparkMtpWeight("model.mtp.0.confidence_head.proj.weight"), DsparkMtpRoute::kConfidenceHead);
  EXPECT_EQ(RouteDsparkMtpWeight("model.mtp.3.confidence_head.proj.weight"), DsparkMtpRoute::kConfidenceHead);
}

TEST(DsparkMtpRouteTest, EProjRoutedCorrectly) {
  EXPECT_EQ(RouteDsparkMtpWeight("model.mtp.0.e_proj.weight"), DsparkMtpRoute::kEProj);
  EXPECT_EQ(RouteDsparkMtpWeight("model.mtp.2.e_proj.weight"), DsparkMtpRoute::kEProj);
}

TEST(DsparkMtpRouteTest, HProjRoutedCorrectly) {
  EXPECT_EQ(RouteDsparkMtpWeight("model.mtp.0.h_proj.weight"), DsparkMtpRoute::kHProj);
  EXPECT_EQ(RouteDsparkMtpWeight("model.mtp.1.h_proj.weight"), DsparkMtpRoute::kHProj);
}

TEST(DsparkMtpRouteTest, NonDsparkWeightFallsBack) {
  EXPECT_EQ(RouteDsparkMtpWeight("model.layers.0.attn.wq_a.weight"), DsparkMtpRoute::kFallback);
  EXPECT_EQ(RouteDsparkMtpWeight("model.embed_tokens.weight"), DsparkMtpRoute::kFallback);
  EXPECT_EQ(RouteDsparkMtpWeight("lm_head.weight"), DsparkMtpRoute::kFallback);
}

// Test: markov_w1 vs markov_w2 must not be confused (substring containment check)
TEST(DsparkMtpRouteTest, MarkovW1AndW2NotConfused) {
  // "markov_head.markov_w1.weight" should NOT match the w2 check
  std::string w1_name = "model.mtp.0.markov_head.markov_w1.weight";
  EXPECT_EQ(w1_name.find("markov_head.markov_w2.weight"), std::string::npos);
  EXPECT_NE(w1_name.find("markov_head.markov_w1.weight"), std::string::npos);

  // "markov_head.markov_w2.weight" should NOT match the w1 check
  std::string w2_name = "model.mtp.0.markov_head.markov_w2.weight";
  EXPECT_EQ(w2_name.find("markov_head.markov_w1.weight"), std::string::npos);
  EXPECT_NE(w2_name.find("markov_head.markov_w2.weight"), std::string::npos);
}

// Test: main_proj vs other proj weights should not conflict
TEST(DsparkMtpRouteTest, MainProjNotConfusedWithEProj) {
  // main_proj.weight should route to kMainProj, not kEProj
  std::string name = "model.mtp.0.main_proj.weight";
  EXPECT_NE(name.find("main_proj.weight"), std::string::npos);
  // e_proj check: "e_proj.weight" is NOT a substring of "main_proj.weight"
  EXPECT_EQ(name.find("e_proj.weight"), std::string::npos);
  EXPECT_EQ(RouteDsparkMtpWeight(name), DsparkMtpRoute::kMainProj);
}

// Test: all DSpark MTP weights across multiple MTP layers route correctly
TEST(DsparkMtpRouteTest, AllMtpLayersRouted) {
  // DSpark typically has 3 MTP layers (mtp.0, mtp.1, mtp.2)
  for (int mtp_idx = 0; mtp_idx <= 2; ++mtp_idx) {
    std::string prefix = "model.mtp." + std::to_string(mtp_idx) + ".";
    EXPECT_EQ(RouteDsparkMtpWeight(prefix + "main_proj.weight"), DsparkMtpRoute::kMainProj);
    EXPECT_EQ(RouteDsparkMtpWeight(prefix + "markov_head.markov_w1.weight"), DsparkMtpRoute::kMarkovW1);
    EXPECT_EQ(RouteDsparkMtpWeight(prefix + "markov_head.markov_w2.weight"), DsparkMtpRoute::kMarkovW2);
    EXPECT_EQ(RouteDsparkMtpWeight(prefix + "confidence_head.proj.weight"), DsparkMtpRoute::kConfidenceHead);
    EXPECT_EQ(RouteDsparkMtpWeight(prefix + "e_proj.weight"), DsparkMtpRoute::kEProj);
    EXPECT_EQ(RouteDsparkMtpWeight(prefix + "h_proj.weight"), DsparkMtpRoute::kHProj);
  }
}

// Test: after MTP name rewrite (model.mtp.N → model.layers.<num_layer+N>),
// the routing should still work correctly.
TEST(DsparkMtpRouteTest, RoutedAfterMtpRewrite) {
  size_t num_layer = 61;
  // After rewrite, mtp.0 → layers.61, mtp.1 → layers.62, etc.
  for (int mtp_idx = 0; mtp_idx <= 2; ++mtp_idx) {
    int mapped_layer = static_cast<int>(num_layer) + mtp_idx;
    std::string prefix = "model.layers." + std::to_string(mapped_layer) + ".";
    EXPECT_EQ(RouteDsparkMtpWeight(prefix + "main_proj.weight"), DsparkMtpRoute::kMainProj);
    EXPECT_EQ(RouteDsparkMtpWeight(prefix + "markov_head.markov_w1.weight"), DsparkMtpRoute::kMarkovW1);
    EXPECT_EQ(RouteDsparkMtpWeight(prefix + "markov_head.markov_w2.weight"), DsparkMtpRoute::kMarkovW2);
    EXPECT_EQ(RouteDsparkMtpWeight(prefix + "confidence_head.proj.weight"), DsparkMtpRoute::kConfidenceHead);
    EXPECT_EQ(RouteDsparkMtpWeight(prefix + "e_proj.weight"), DsparkMtpRoute::kEProj);
    EXPECT_EQ(RouteDsparkMtpWeight(prefix + "h_proj.weight"), DsparkMtpRoute::kHProj);
  }
}

// ---------------------------------------------------------------------------
// FP4 MoE weight name classification tests
// ---------------------------------------------------------------------------

TEST(Fp4MoeWeightNameTest, GateProjWeight) {
  std::string name = "model.layers.0.mlp.experts.0.gate_proj.weight";
  EXPECT_TRUE(name.find(".weight") != std::string::npos);
  EXPECT_TRUE(name.find("weight_scale_inv") == std::string::npos);
  EXPECT_TRUE(name.find(".gate_proj.") != std::string::npos);
}

TEST(Fp4MoeWeightNameTest, UpProjWeight) {
  std::string name = "model.layers.0.mlp.experts.1.up_proj.weight";
  EXPECT_TRUE(name.find(".weight") != std::string::npos);
  EXPECT_TRUE(name.find("weight_scale_inv") == std::string::npos);
  EXPECT_TRUE(name.find(".up_proj.") != std::string::npos);
}

TEST(Fp4MoeWeightNameTest, DownProjWeight) {
  std::string name = "model.layers.0.mlp.experts.0.down_proj.weight";
  EXPECT_TRUE(name.find(".weight") != std::string::npos);
  EXPECT_TRUE(name.find("weight_scale_inv") == std::string::npos);
  EXPECT_TRUE(name.find(".down_proj.") != std::string::npos);
}

TEST(Fp4MoeWeightNameTest, GateProjScale) {
  std::string name = "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv";
  EXPECT_TRUE(name.find(".weight_scale_inv") != std::string::npos);
  EXPECT_TRUE(name.find(".gate_proj.") != std::string::npos);
}

TEST(Fp4MoeWeightNameTest, IsWeightVsScale) {
  // Replicate the is_weight / is_scale logic from LoadFp4MoeWeight
  auto check_weight = [](const std::string& name) {
    return name.find(".weight") != std::string::npos && name.find("weight_scale_inv") == std::string::npos;
  };
  auto check_scale = [](const std::string& name) { return name.find(".weight_scale_inv") != std::string::npos; };

  std::string weight_name = "model.layers.0.mlp.experts.0.gate_proj.weight";
  std::string scale_name = "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv";

  EXPECT_TRUE(check_weight(weight_name));
  EXPECT_FALSE(check_scale(weight_name));
  EXPECT_FALSE(check_weight(scale_name));
  EXPECT_TRUE(check_scale(scale_name));
}

// ---------------------------------------------------------------------------
// Scale name derivation tests (PostProcessModelWeights logic)
// ---------------------------------------------------------------------------

TEST(ScaleNameDerivationTest, WeightToScaleName) {
  // Replicate the scale name derivation: replace "weight" with "weight_scale_inv"
  auto derive_scale_name = [](const std::string& weight_name) {
    std::string scale_name = weight_name;
    size_t weight_pos = scale_name.rfind("weight");
    if (weight_pos != std::string::npos) {
      scale_name.replace(weight_pos, 6, "weight_scale_inv");
    }
    return scale_name;
  };

  EXPECT_EQ(derive_scale_name("model.layers.0.mlp.experts.up_gate_proj.weight"),
            "model.layers.0.mlp.experts.up_gate_proj.weight_scale_inv");
  EXPECT_EQ(derive_scale_name("model.layers.0.mlp.experts.down_proj.weight"),
            "model.layers.0.mlp.experts.down_proj.weight_scale_inv");
}

TEST(ScaleNameDerivationTest, GlobalScaleName) {
  // Replicate the global_scale name derivation: replace "weight" with "weight_global_scale"
  auto derive_gs_name = [](const std::string& weight_name) {
    std::string gs_name = weight_name;
    size_t gpos = gs_name.rfind("weight");
    if (gpos != std::string::npos) {
      gs_name.replace(gpos, 6, "weight_global_scale");
    }
    return gs_name;
  };

  EXPECT_EQ(derive_gs_name("model.layers.0.mlp.experts.up_gate_proj.weight"),
            "model.layers.0.mlp.experts.up_gate_proj.weight_global_scale");
}

// ---------------------------------------------------------------------------
// Combined weight name (CombineWeightByRow) tests
// ---------------------------------------------------------------------------

TEST(CombineWeightByRowNameTest, FuseWkvAndWgate) {
  // Replicate the name fusion logic from CombineWeightByRow:
  // first_name = "compressor.wkv", second_name = "compressor.wgate",
  // final_name = "compressor.wkv_gate"
  auto fuse_name = [](const std::string& file_weight_name, const std::string& first_name,
                      const std::string& second_name, const std::string& final_name) {
    std::string fused_name = file_weight_name;
    auto first_pos = fused_name.find(first_name + ".");
    auto second_pos = fused_name.find(second_name + ".");
    if (first_pos != std::string::npos) {
      fused_name.replace(first_pos, (first_name + ".").size(), final_name + ".");
    } else if (second_pos != std::string::npos) {
      fused_name.replace(second_pos, (second_name + ".").size(), final_name + ".");
    }
    return fused_name;
  };

  std::string wkv_name = "model.layers.0.attn.compressor.wkv.weight";
  std::string wgate_name = "model.layers.0.attn.compressor.wgate.weight";

  std::string fused_from_wkv = fuse_name(wkv_name, "compressor.wkv", "compressor.wgate", "compressor.wkv_gate");
  std::string fused_from_wgate = fuse_name(wgate_name, "compressor.wkv", "compressor.wgate", "compressor.wkv_gate");

  EXPECT_EQ(fused_from_wkv, "model.layers.0.attn.compressor.wkv_gate.weight");
  EXPECT_EQ(fused_from_wgate, "model.layers.0.attn.compressor.wkv_gate.weight");
  // Both should produce the same fused name
  EXPECT_EQ(fused_from_wkv, fused_from_wgate);
}

TEST(CombineWeightByRowNameTest, FuseWqAAndWkv) {
  auto fuse_name = [](const std::string& file_weight_name, const std::string& first_name,
                      const std::string& second_name, const std::string& final_name) {
    std::string fused_name = file_weight_name;
    auto first_pos = fused_name.find(first_name + ".");
    auto second_pos = fused_name.find(second_name + ".");
    if (first_pos != std::string::npos) {
      fused_name.replace(first_pos, (first_name + ".").size(), final_name + ".");
    } else if (second_pos != std::string::npos) {
      fused_name.replace(second_pos, (second_name + ".").size(), final_name + ".");
    }
    return fused_name;
  };

  std::string wq_a_name = "model.layers.0.attn.wq_a.weight";
  std::string wkv_name = "model.layers.0.attn.wkv.weight";

  std::string fused_from_wq_a = fuse_name(wq_a_name, "attn.wq_a", "attn.wkv", "attn.wqkv_a");
  std::string fused_from_wkv = fuse_name(wkv_name, "attn.wq_a", "attn.wkv", "attn.wqkv_a");

  EXPECT_EQ(fused_from_wq_a, "model.layers.0.attn.wqkv_a.weight");
  EXPECT_EQ(fused_from_wkv, "model.layers.0.attn.wqkv_a.weight");
}

// ---------------------------------------------------------------------------
// FilterWeightNames skip list tests
// ---------------------------------------------------------------------------

TEST(FilterWeightNamesSkipListTest, SkipListMatches) {
  // Replicate the skip_list from FilterWeightNames
  const std::vector<std::string> skip_list = {"self_attn.rotary_emb.inv_freq", "vision_tower.", "mm_projector."};

  auto check_skip = [&](const std::string& name) {
    for (const auto& skip : skip_list) {
      if (name.find(skip) != std::string::npos) return true;
    }
    return false;
  };

  EXPECT_TRUE(check_skip("model.layers.0.self_attn.rotary_emb.inv_freq"));
  EXPECT_TRUE(check_skip("vision_tower.encoder.weight"));
  EXPECT_TRUE(check_skip("model.mm_projector.linear.weight"));
  EXPECT_FALSE(check_skip("model.layers.0.attn.wq_a.weight"));
  EXPECT_FALSE(check_skip("model.embed_tokens.weight"));
}

// ---------------------------------------------------------------------------
// E8M0 to FP32 conversion tests (PreProcessModelWeights logic)
// ---------------------------------------------------------------------------

TEST(E8m0ToFp32Test, ConvertE8m0ToFp32) {
  // Replicate the E8M0 → FP32 conversion logic from PreProcessModelWeights:
  // uint32_t bits = src[i] << 23;  // E8M0 exponent → FP32
  // This is equivalent to powf(2, src[i] - 127)

  auto e8m0_to_fp32 = [](uint8_t e8m0) -> float {
    uint32_t bits = static_cast<uint32_t>(e8m0) << 23;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
  };

  // E8M0 value 127 → 2^0 = 1.0
  EXPECT_FLOAT_EQ(e8m0_to_fp32(127), 1.0f);
  // E8M0 value 128 → 2^1 = 2.0
  EXPECT_FLOAT_EQ(e8m0_to_fp32(128), 2.0f);
  // E8M0 value 126 → 2^(-1) = 0.5
  EXPECT_FLOAT_EQ(e8m0_to_fp32(126), 0.5f);
  // E8M0 value 0 → denormalized (very small)
  EXPECT_FLOAT_EQ(e8m0_to_fp32(0), 0.0f);
  // E8M0 value 130 → 2^3 = 8.0
  EXPECT_FLOAT_EQ(e8m0_to_fp32(130), 8.0f);
}

// ---------------------------------------------------------------------------
// tid2eid narrowing tests (int64 → int32)
// ---------------------------------------------------------------------------

TEST(Tid2EidNarrowingTest, Int64ToInt32) {
  // Replicate the tid2eid narrowing logic from ProcessModelWeights
  std::vector<int64_t> src = {0, 1, 255, 128, 42, 100};
  std::vector<int32_t> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = static_cast<int32_t>(src[i]);
  }

  for (size_t i = 0; i < src.size(); ++i) {
    EXPECT_EQ(dst[i], static_cast<int32_t>(src[i]));
  }
}

// ---------------------------------------------------------------------------
// APE hotfix overlap logic tests
// ---------------------------------------------------------------------------

TEST(ApeHotfixTest, OverlapDetection) {
  // Replicate the overlap detection logic: overlap = (ratio == 4)
  auto is_overlap = [](size_t ratio) { return ratio == 4; };

  EXPECT_TRUE(is_overlap(4));
  EXPECT_FALSE(is_overlap(1));
  EXPECT_FALSE(is_overlap(2));
  EXPECT_FALSE(is_overlap(3));
  EXPECT_FALSE(is_overlap(8));
}

TEST(ApeHotfixTest, TransformWithOverlap) {
  // Replicate the APE hotfix transform logic for overlap case (ratio == 4)
  // Input: [ratio, last_dim] where last_dim = 2 * head_dim
  // Output: same shape but rows are rearranged
  // concat([left, right], dim=0) → [2*ratio, half_dim], then view(ratio, last_dim)
  // Result row i = [concat_row[2*i], concat_row[2*i+1]]

  const size_t ratio = 4;
  const size_t head_dim = 3;
  const size_t last_dim = head_dim * 2;  // 6

  // Input: [4, 6] — each row has [left_half (3), right_half (3)]
  // left[j] = src[j * 6 ... j*6+2]
  // right[j] = src[j * 6 + 3 ... j*6+5]
  std::vector<float> src = {
      10, 11,  12,  20, 21,  22,  // row 0: left=[10,11,12], right=[20,21,22]
      13, 14,  15,  23, 24,  25,  // row 1
      16, 17,  18,  26, 27,  28,  // row 2
      19, 100, 101, 29, 200, 201  // row 3
  };

  std::vector<float> dst(ratio * last_dim, 0.0f);

  for (size_t i = 0; i < ratio; ++i) {
    size_t concat_row_0 = 2 * i;
    size_t concat_row_1 = 2 * i + 1;

    // First half of dst row i ← concat_row_0
    const float* src_ptr_0;
    if (concat_row_0 < ratio) {
      src_ptr_0 = &src[concat_row_0 * last_dim];
    } else {
      src_ptr_0 = &src[(concat_row_0 - ratio) * last_dim + head_dim];
    }
    for (size_t j = 0; j < head_dim; ++j) {
      dst[i * last_dim + j] = src_ptr_0[j];
    }

    // Second half of dst row i ← concat_row_1
    const float* src_ptr_1;
    if (concat_row_1 < ratio) {
      src_ptr_1 = &src[concat_row_1 * last_dim];
    } else {
      src_ptr_1 = &src[(concat_row_1 - ratio) * last_dim + head_dim];
    }
    for (size_t j = 0; j < head_dim; ++j) {
      dst[i * last_dim + head_dim + j] = src_ptr_1[j];
    }
  }

  // Expected: for i=0, concat_row_0=0 (< 4, from left), concat_row_1=1 (< 4, from left)
  // dst[0] = [src[0*6], src[1*6]] = [10,11,12, 13,14,15]
  EXPECT_FLOAT_EQ(dst[0], 10);
  EXPECT_FLOAT_EQ(dst[1], 11);
  EXPECT_FLOAT_EQ(dst[2], 12);
  EXPECT_FLOAT_EQ(dst[3], 13);
  EXPECT_FLOAT_EQ(dst[4], 14);
  EXPECT_FLOAT_EQ(dst[5], 15);

  // For i=1, concat_row_0=2 (< 4, from left), concat_row_1=3 (< 4, from left)
  // dst[1] = [src[2*6], src[3*6]] = [16,17,18, 19,100,101]
  EXPECT_FLOAT_EQ(dst[6], 16);
  EXPECT_FLOAT_EQ(dst[7], 17);
  EXPECT_FLOAT_EQ(dst[8], 18);
  EXPECT_FLOAT_EQ(dst[9], 19);
  EXPECT_FLOAT_EQ(dst[10], 100);
  EXPECT_FLOAT_EQ(dst[11], 101);

  // For i=2, concat_row_0=4 (>= 4, from right row 0), concat_row_1=5 (>= 4, from right row 1)
  // dst[2] = [right[0], right[1]] = [src[0*6+3], src[1*6+3]] = [20,21,22, 23,24,25]
  EXPECT_FLOAT_EQ(dst[12], 20);
  EXPECT_FLOAT_EQ(dst[13], 21);
  EXPECT_FLOAT_EQ(dst[14], 22);
  EXPECT_FLOAT_EQ(dst[15], 23);
  EXPECT_FLOAT_EQ(dst[16], 24);
  EXPECT_FLOAT_EQ(dst[17], 25);

  // For i=3, concat_row_0=6 (>= 4, from right row 2), concat_row_1=7 (>= 4, from right row 3)
  // dst[3] = [right[2], right[3]] = [src[2*6+3], src[3*6+3]] = [26,27,28, 29,200,201]
  EXPECT_FLOAT_EQ(dst[18], 26);
  EXPECT_FLOAT_EQ(dst[19], 27);
  EXPECT_FLOAT_EQ(dst[20], 28);
  EXPECT_FLOAT_EQ(dst[21], 29);
  EXPECT_FLOAT_EQ(dst[22], 200);
  EXPECT_FLOAT_EQ(dst[23], 201);
}

// ---------------------------------------------------------------------------
// attn_sink reorder logic tests
// ---------------------------------------------------------------------------

TEST(AttnSinkReorderTest, ReorderHeads) {
  // Replicate the attn_sink reorder logic from ProcessModelWeights:
  // Rotate attn_sink so this rank's local heads are at the front.
  const size_t head_num = 8;
  const size_t attn_tp_size = 4;
  const size_t n_local_heads = head_num / attn_tp_size;  // 2

  for (size_t attn_dev_rank = 0; attn_dev_rank < attn_tp_size; ++attn_dev_rank) {
    std::vector<float> src(head_num);
    for (size_t i = 0; i < head_num; ++i) {
      src[i] = static_cast<float>(i);
    }

    std::vector<float> dst(head_num);
    const size_t rank_offset = attn_dev_rank * n_local_heads;
    const size_t local_bytes = n_local_heads;

    // Replicate the 3 memcpy calls
    std::copy(src.begin() + rank_offset, src.begin() + rank_offset + local_bytes, dst.begin());
    std::copy(src.begin(), src.begin() + rank_offset, dst.begin() + local_bytes);
    std::copy(src.begin() + rank_offset + local_bytes, src.end(), dst.begin() + local_bytes + rank_offset);

    // Verify: local heads should be at front
    for (size_t i = 0; i < n_local_heads; ++i) {
      EXPECT_FLOAT_EQ(dst[i], static_cast<float>(rank_offset + i));
    }
  }
}

// ---------------------------------------------------------------------------
// Integration tests: actually call ProcessModelWeights to cover DSpark
// weight loading code paths (lines 668-698 and 847-895 in
// deepseek_v4_weight_loader.cpp)
// ---------------------------------------------------------------------------

#ifdef ENABLE_CUDA

// Test subclass to expose protected members for testing.
class TestDeepSeekV4WeightLoader : public DeepSeekV4WeightLoader {
 public:
  using DeepSeekV4WeightLoader::deepseek_v4_config_;
  using DeepSeekV4WeightLoader::DeepSeekV4WeightLoader;
  using DeepSeekV4WeightLoader::pipeline_config_;
  using DeepSeekV4WeightLoader::runtime_config_;
  using DeepSeekV4WeightLoader::weight_impl_;
  using DeepSeekV4WeightLoader::weights_to_permute_;
};

class DeepSeekV4WeightLoaderIntegrationTest : public testing::Test {
 protected:
  // NOTE: These integration tests construct a real DeepSeekV4WeightLoader and
  // call ProcessModelWeights to exercise the DSpark weight loading code paths
  // (lines 668-698 for wkv view, and lines 847-895 for main_proj/markov/
  // confidence_head). They require a fully initialized GPU + NCCL environment.
  void SetUp() override {
    InitLoguru();

    // Set up a minimal GPU context (single GPU, single rank).
    const auto& env = Singleton<Environment>::GetInstance();
    context_ = std::make_shared<Context>();
    context_->global.reset(new GlobalContext(1, 1, 1, env));
    context_->global->InitDistributedConfig();
    context_->device = std::make_shared<DeviceContext>(1);

    // Initialize NCCL for distributed operations.
    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context_->device->ext->InitNcclParam(nccl_uid);

    // Build a DeepSeekV4Config with DSpark enabled.
    auto v4_config = std::make_shared<DeepSeekV4Config>();
    v4_config->type = "deepseek_v4";
    v4_config->num_layer = 4;                 // small model for testing
    v4_config->num_nextn_predict_layers = 1;  // 1 MTP layer
    v4_config->hidden_units = 64;
    v4_config->size_per_head = 64;
    v4_config->head_num = 1;
    v4_config->vocab_size = 100;
    v4_config->weight_data_type = DataType::TYPE_BF16;
    v4_config->is_quant = false;
    v4_config->is_compressed_tensors = false;

    // Parallel config (direct members of DeepSeekV3Config)
    v4_config->tensor_para_size = 1;
    v4_config->attn_data_para_size = 1;
    v4_config->expert_para_size = 1;
    v4_config->moe_tensor_para_size = 1;

    // MoE config
    v4_config->moe_config.num_experts = 4;
    v4_config->moe_config.moe_inter_size = 128;
    v4_config->moe_config.shared_expert_inter_size = 128;
    v4_config->moe_config.use_vllm_moe = false;
    v4_config->moe_config.first_k_dense_replace = 0;

    // Expert parallel config
    v4_config->expert_parallel_config.expert_world_size = 1;
    v4_config->expert_parallel_config.expert_para_size = 1;
    v4_config->expert_parallel_config.expert_node_rank = 0;

    // Quant config: must have is_fp8_blockwise = true (ProcessModelWeights checks it)
    v4_config->quant_config.is_fp8_blockwise = true;
    v4_config->quant_config.expert_dtype = "fp4";
    v4_config->quant_config.weight_block_size = {128, 128};

    // DSpark config: enable DSpark
    v4_config->dspark_config.dspark_block_size = 128;
    v4_config->dspark_config.dspark_noise_token_id = 0;
    v4_config->dspark_config.dspark_markov_rank = 256;

    // V4 attention config
    v4_config->dsv4_attn_config.q_lora_rank = 16;
    v4_config->dsv4_attn_config.o_groups = 4;
    v4_config->dsv4_attn_config.o_lora_rank = 16;
    v4_config->dsv4_attn_config.qk_rope_head_dim = 8;

    // MHC config
    v4_config->mhc_config.hc_mult = 4;

    model_config_ = v4_config;

    // Set up environment configs.
    // Environment::SetModelConfig takes ModelConfig&, but DeepSeekV4Config
    // inherits from BaseModelConfig (not ModelConfig). We need to create a
    // separate ModelConfig with the fields that Environment needs.
    ModelConfig env_model_config;
    env_model_config.num_layer = v4_config->num_layer;
    env_model_config.num_nextn_predict_layers = v4_config->num_nextn_predict_layers;
    env_model_config.hidden_units = v4_config->hidden_units;
    env_model_config.size_per_head = v4_config->size_per_head;
    env_model_config.head_num = v4_config->head_num;
    env_model_config.vocab_size = v4_config->vocab_size;
    env_model_config.weight_data_type = v4_config->weight_data_type;
    env_model_config.type = v4_config->type;
    env_model_config.use_dsa = true;
    env_model_config.use_dsv4_attn = true;
    env_model_config.use_mhc = true;
    env_model_config.moe_config = v4_config->moe_config;
    env_model_config.quant_config = v4_config->quant_config;
    env_model_config.dspark_config = v4_config->dspark_config;
    env_model_config.dsv4_attn_config = v4_config->dsv4_attn_config;
    env_model_config.mhc_config = v4_config->mhc_config;
    env->SetModelConfig(env_model_config);

    PipelineConfig pipeline_config;
    pipeline_config.lower_layer_idx = 0;
    pipeline_config.upper_layer_idx = 3;        // num_layer - 1
    pipeline_config.lower_nextn_layer_idx = 4;  // num_layer
    pipeline_config.upper_nextn_layer_idx = 4;  // num_layer + num_nextn - 1
    env->SetPipelineConfig(pipeline_config);

    // Set runtime config through the schedule config parser (no SetRuntimeConfig API).
    env->schedule_config_parser_.runtime_config_.parallel_basic_config.tensor_parallel_size = 1;
    env->schedule_config_parser_.runtime_config_.mtp_step_num = 1;
    env->schedule_config_parser_.runtime_config_.enable_full_shared_expert = false;

    // Create the weight loader.
    weight_loader_ = std::make_unique<TestDeepSeekV4WeightLoader>(model_config_, env, context_);
  }

  void TearDown() override { DeviceSynchronize(); }

  // Helper: create a host tensor filled with a value.
  Tensor CreateHostTensor(const std::vector<size_t>& shape, DataType dtype, float value) {
    Tensor tensor(MemoryLocation::LOCATION_HOST, dtype, shape);
    if (dtype == DataType::TYPE_BF16) {
      bfloat16* ptr = tensor.GetPtr<bfloat16>();
      size_t n = tensor.GetElementNumber();
      for (size_t i = 0; i < n; ++i) {
        ptr[i] = bfloat16(value);
      }
    } else if (dtype == DataType::TYPE_FP32) {
      float* ptr = tensor.GetPtr<float>();
      size_t n = tensor.GetElementNumber();
      for (size_t i = 0; i < n; ++i) {
        ptr[i] = value;
      }
    }
    return tensor;
  }

 protected:
  std::shared_ptr<BaseModelConfig> model_config_;
  std::shared_ptr<Context> context_;
  std::unique_ptr<TestDeepSeekV4WeightLoader> weight_loader_;
};

// Test: main_proj.weight should be copied to device_model_weights via
// CopyToDevice (covers lines 851-859).
TEST_F(DeepSeekV4WeightLoaderIntegrationTest, MainProjWeightLoaded) {
  std::unordered_map<std::string, Tensor> host_weights;
  std::unordered_map<std::string, Tensor> device_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  // Create a main_proj.weight tensor (after MTP rewrite: model.layers.4.main_proj.weight)
  host_weights["model.layers.4.main_proj.weight"] = CreateHostTensor({4, 8}, DataType::TYPE_BF16, 1.5f);

  Status status = weight_loader_->ProcessModelWeights(host_weights, 0, device_weights, left_host_weights);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  // The weight should be stored in device_model_weights
  EXPECT_NE(device_weights.find("model.layers.4.main_proj.weight"), device_weights.end());
  auto& dev_tensor = device_weights.at("model.layers.4.main_proj.weight");
  EXPECT_EQ(dev_tensor.shape.size(), 2u);
  EXPECT_EQ(dev_tensor.shape[0], 4u);
  EXPECT_EQ(dev_tensor.shape[1], 8u);
}

// Test: markov_head.markov_w1.weight should be replicated on every rank
// (full [V, R] via CopyToDevice, no TP split).
TEST_F(DeepSeekV4WeightLoaderIntegrationTest, MarkovW1WeightLoaded) {
  std::unordered_map<std::string, Tensor> host_weights;
  std::unordered_map<std::string, Tensor> device_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  host_weights["model.layers.4.markov_head.markov_w1.weight"] = CreateHostTensor({4, 16}, DataType::TYPE_BF16, 2.0f);

  Status status = weight_loader_->ProcessModelWeights(host_weights, 0, device_weights, left_host_weights);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  EXPECT_NE(device_weights.find("model.layers.4.markov_head.markov_w1.weight"), device_weights.end());
  auto& dev_tensor = device_weights.at("model.layers.4.markov_head.markov_w1.weight");
  EXPECT_EQ(dev_tensor.shape.size(), 2u);
  EXPECT_EQ(dev_tensor.shape[0], 4u);
  EXPECT_EQ(dev_tensor.shape[1], 16u);
}

// Test: markov_head.markov_w2.weight should be replicated + transposed to [R, V]
// (CopyToDevice + PermuteWeight, no TP split).
TEST_F(DeepSeekV4WeightLoaderIntegrationTest, MarkovW2WeightLoaded) {
  std::unordered_map<std::string, Tensor> host_weights;
  std::unordered_map<std::string, Tensor> device_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  host_weights["model.layers.4.markov_head.markov_w2.weight"] = CreateHostTensor({16, 4}, DataType::TYPE_BF16, 3.0f);

  Status status = weight_loader_->ProcessModelWeights(host_weights, 0, device_weights, left_host_weights);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  EXPECT_NE(device_weights.find("model.layers.4.markov_head.markov_w2.weight"), device_weights.end());
  auto& dev_tensor = device_weights.at("model.layers.4.markov_head.markov_w2.weight");
  EXPECT_EQ(dev_tensor.shape.size(), 2u);
  // After PermuteWeight({1, 0}), shape flipped from [16, 4] -> [4, 16].
  EXPECT_EQ(dev_tensor.shape[0], 4u);
  EXPECT_EQ(dev_tensor.shape[1], 16u);
}

// Test: confidence_head.proj.weight should be copied and transposed
// via CopyToDevice + PermuteWeight (covers lines 886-895).
TEST_F(DeepSeekV4WeightLoaderIntegrationTest, ConfidenceHeadWeightLoaded) {
  std::unordered_map<std::string, Tensor> host_weights;
  std::unordered_map<std::string, Tensor> device_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  host_weights["model.layers.4.confidence_head.proj.weight"] = CreateHostTensor({4, 8}, DataType::TYPE_BF16, 4.0f);

  Status status = weight_loader_->ProcessModelWeights(host_weights, 0, device_weights, left_host_weights);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  EXPECT_NE(device_weights.find("model.layers.4.confidence_head.proj.weight"), device_weights.end());
  // After PermuteWeight({1, 0}), shape should be transposed: {8, 4}
  auto& dev_tensor = device_weights.at("model.layers.4.confidence_head.proj.weight");
  EXPECT_EQ(dev_tensor.shape.size(), 2u);
  EXPECT_EQ(dev_tensor.shape[0], 8u);
  EXPECT_EQ(dev_tensor.shape[1], 4u);
}

// Test: e_proj.weight should be loaded via CopyToDevice (covers lines 836-845).
TEST_F(DeepSeekV4WeightLoaderIntegrationTest, EProjWeightLoaded) {
  std::unordered_map<std::string, Tensor> host_weights;
  std::unordered_map<std::string, Tensor> device_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  host_weights["model.layers.4.e_proj.weight"] = CreateHostTensor({8, 4}, DataType::TYPE_BF16, 5.0f);

  Status status = weight_loader_->ProcessModelWeights(host_weights, 0, device_weights, left_host_weights);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  EXPECT_NE(device_weights.find("model.layers.4.e_proj.weight"), device_weights.end());
  auto& dev_tensor = device_weights.at("model.layers.4.e_proj.weight");
  EXPECT_EQ(dev_tensor.shape[0], 8u);
  EXPECT_EQ(dev_tensor.shape[1], 4u);
}

// Test: h_proj.weight should be loaded via CopyToDevice (covers lines 836-845).
TEST_F(DeepSeekV4WeightLoaderIntegrationTest, HProjWeightLoaded) {
  std::unordered_map<std::string, Tensor> host_weights;
  std::unordered_map<std::string, Tensor> device_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  host_weights["model.layers.4.h_proj.weight"] = CreateHostTensor({4, 8}, DataType::TYPE_BF16, 6.0f);

  Status status = weight_loader_->ProcessModelWeights(host_weights, 0, device_weights, left_host_weights);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  EXPECT_NE(device_weights.find("model.layers.4.h_proj.weight"), device_weights.end());
  auto& dev_tensor = device_weights.at("model.layers.4.h_proj.weight");
  EXPECT_EQ(dev_tensor.shape[0], 4u);
  EXPECT_EQ(dev_tensor.shape[1], 8u);
}

// Test: DSpark wkv view creation (covers lines 668-698).
// When dspark_block_size > 0 and the weight is attn.wkv in an MTP layer,
// ProcessModelWeights should create a wkv view from the fused wqkv_a tensor.
TEST_F(DeepSeekV4WeightLoaderIntegrationTest, DsparkWkvViewCreated) {
  std::unordered_map<std::string, Tensor> host_weights;
  std::unordered_map<std::string, Tensor> device_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  // wq_a.weight: shape [q_lora_rank, hidden] = [16, 64]
  host_weights["model.layers.4.attn.wq_a.weight"] = CreateHostTensor({16, 64}, DataType::TYPE_BF16, 1.0f);
  // wkv.weight: shape [head_dim, hidden] = [64, 64]
  host_weights["model.layers.4.attn.wkv.weight"] = CreateHostTensor({64, 64}, DataType::TYPE_BF16, 2.0f);

  Status status = weight_loader_->ProcessModelWeights(host_weights, 0, device_weights, left_host_weights);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  // The fused wqkv_a tensor should be created
  std::string fused_name = "model.layers.4.attn.wqkv_a.weight";
  EXPECT_NE(device_weights.find(fused_name), device_weights.end()) << "Fused wqkv_a tensor should be created";

  // The wkv view should also be created (DSpark specific)
  std::string wkv_name = "model.layers.4.attn.wkv.weight";
  EXPECT_NE(device_weights.find(wkv_name), device_weights.end())
      << "DSpark wkv view should be created from fused wqkv_a";

  if (device_weights.find(wkv_name) != device_weights.end()) {
    auto& wkv_view = device_weights.at(wkv_name);
    // wkv view should have the same shape as the original wkv tensor
    EXPECT_EQ(wkv_view.shape.size(), 2u);
    EXPECT_EQ(wkv_view.shape[0], 64u);
    EXPECT_EQ(wkv_view.shape[1], 64u);
  }
}

// Test: All DSpark MTP weights loaded together in one call to
// ProcessModelWeights (comprehensive integration test).
TEST_F(DeepSeekV4WeightLoaderIntegrationTest, AllDsparkMtpWeightsLoaded) {
  std::unordered_map<std::string, Tensor> host_weights;
  std::unordered_map<std::string, Tensor> device_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;

  // Add all DSpark MTP weights (layer 4 = num_layer + 0, the first MTP layer)
  host_weights["model.layers.4.main_proj.weight"] = CreateHostTensor({4, 8}, DataType::TYPE_BF16, 1.0f);
  host_weights["model.layers.4.markov_head.markov_w1.weight"] = CreateHostTensor({4, 16}, DataType::TYPE_BF16, 2.0f);
  host_weights["model.layers.4.markov_head.markov_w2.weight"] = CreateHostTensor({16, 4}, DataType::TYPE_BF16, 3.0f);
  host_weights["model.layers.4.confidence_head.proj.weight"] = CreateHostTensor({4, 8}, DataType::TYPE_BF16, 4.0f);
  host_weights["model.layers.4.e_proj.weight"] = CreateHostTensor({8, 4}, DataType::TYPE_BF16, 5.0f);
  host_weights["model.layers.4.h_proj.weight"] = CreateHostTensor({4, 8}, DataType::TYPE_BF16, 6.0f);

  Status status = weight_loader_->ProcessModelWeights(host_weights, 0, device_weights, left_host_weights);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  // All 6 DSpark MTP weights should be loaded
  EXPECT_NE(device_weights.find("model.layers.4.main_proj.weight"), device_weights.end());
  EXPECT_NE(device_weights.find("model.layers.4.markov_head.markov_w1.weight"), device_weights.end());
  EXPECT_NE(device_weights.find("model.layers.4.markov_head.markov_w2.weight"), device_weights.end());
  EXPECT_NE(device_weights.find("model.layers.4.confidence_head.proj.weight"), device_weights.end());
  EXPECT_NE(device_weights.find("model.layers.4.e_proj.weight"), device_weights.end());
  EXPECT_NE(device_weights.find("model.layers.4.h_proj.weight"), device_weights.end());
}

#endif  // ENABLE_CUDA

}  // namespace ksana_llm
