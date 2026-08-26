/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "configure/json_config_utils.h"
#include "model_loader/config_parser/model_config_parser_factory.h"
#include "model_loader/weight_loader/model_weight_loader_factory.h"
#include "models/base/forwarding_context.h"
#include "models/base/model_arch.h"
#include "models/hy_v3/hy_v3_config.h"
#include "models/hy_v3/hy_v3_config_parser.h"
#include "models/hy_v3/hy_v3_model.h"
#include "models/hy_v3/hy_v3_mtp_draft_model.h"
#include "models/hy_v3/hy_v3_weight_loader.h"
#include "nlohmann/json.hpp"
#include "test.h"
#include "utils/tensor.h"

#ifdef ENABLE_CUDA
#  include "configure/context.h"
#  include "configure/environment.h"
#  include "device/nvidia/nccl_utils.h"
#  include "helpers/block_cache_layout_helper.h"
#  include "helpers/model_test_helper.h"
#  include "models/base/base_weight.h"
#  include "utils/attention_backend/attention_backend_manager.h"
#  include "utils/singleton.h"
#endif

using namespace ksana_llm;

namespace {

// A minimal config.json mirroring /data1/models/Hy3-29183/config.json.
nlohmann::json MakeHyV3Config() {
  return nlohmann::json{
      {"architectures", {"HYV3ForCausalLM"}},
      {"model_type", "hy_v3"},
      {"torch_dtype", "bfloat16"},
      {"bos_token_id", 120000},
      {"eos_token_id", 120025},
      {"pad_token_id", 120002},
      {"hidden_size", 4096},
      {"intermediate_size", 13312},
      {"head_dim", 128},
      {"num_attention_heads", 64},
      {"num_key_value_heads", 8},
      {"num_hidden_layers", 80},
      {"vocab_size", 120832},
      {"rms_norm_eps", 1e-05},
      {"max_position_embeddings", 262144},
      {"qk_norm", true},
      {"tie_word_embeddings", false},
      {"num_experts", 192},
      {"num_experts_per_tok", 8},
      {"expert_hidden_dim", 1536},
      {"moe_intermediate_size", 1536},
      {"num_shared_experts", 1},
      {"first_k_dense_replace", 1},
      {"moe_router_use_sigmoid", true},
      {"moe_router_enable_expert_bias", true},
      {"route_norm", true},
      {"router_scaling_factor", 2.826},
      {"num_nextn_predict_layers", 1},
      {"rope_parameters", {{"rope_theta", 11158840.0}, {"rope_type", "default"}}},
  };
}

class HyV3ConfigParserTest : public testing::Test {
 protected:
  void SetUp() override { InitLoguru(); }
  HyV3ConfigParser parser_;
  ParallelismBasicConfig parallel_config_;
};

TEST_F(HyV3ConfigParserTest, ParsesArchitectureFields) {
  nlohmann::json config_json = MakeHyV3Config();
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3", base_config).OK());

  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);

  // Attention.
  EXPECT_EQ(config->head_num, 64u);
  EXPECT_EQ(config->num_key_value_heads, 8u);
  EXPECT_EQ(config->size_per_head, 128u);
  EXPECT_EQ(config->rotary_embedding, 128u);  // full rotary, no partial factor
  EXPECT_EQ(config->hidden_units, 4096u);
  EXPECT_EQ(config->inter_size, 13312u);
  EXPECT_EQ(config->vocab_size, 120832u);
  EXPECT_EQ(config->num_layer, 80u);
  EXPECT_TRUE(config->use_qk_norm);
  EXPECT_FALSE(config->enable_add_qkv_bias);  // hy_v3 has no QKV bias

  // rope_theta must be read from the nested rope_parameters object.
  EXPECT_FLOAT_EQ(config->rope_theta, 11158840.0f);

  // MTP (num_nextn_predict_layers) is honored from config.json.
  EXPECT_EQ(config->num_nextn_predict_layers, 1u);
}

TEST_F(HyV3ConfigParserTest, ParsesMoeGatingFields) {
  nlohmann::json config_json = MakeHyV3Config();
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3", base_config).OK());
  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);

  EXPECT_TRUE(config->is_moe);
  EXPECT_EQ(config->moe_config.num_experts, 192u);
  EXPECT_EQ(config->moe_config.experts_topk, 8u);
  EXPECT_EQ(config->moe_config.moe_inter_size, 1536u);
  EXPECT_EQ(config->moe_config.first_k_dense_replace, 1u);
  EXPECT_EQ(config->moe_config.num_expert_group, 1u);
  EXPECT_EQ(config->moe_config.expert_groups_topk, 1u);
  EXPECT_EQ(config->moe_config.scoring_func, "sigmoid");
  EXPECT_TRUE(config->moe_config.norm_topk_prob);
  EXPECT_TRUE(config->moe_config.use_e_score_correction_bias);
  EXPECT_FLOAT_EQ(config->moe_config.routed_scaling_factor, 2.826f);
  // Must match PrepareHyV3Attributes: (fast, sigmoid) is the only pairing Fp8MoeLayer wires.
  EXPECT_EQ(config->moe_config.topk_method, "fast");

  // Single merged shared expert.
  EXPECT_TRUE(config->has_shared_experts);
  EXPECT_EQ(config->moe_config.num_shared_experts, 1u);
  EXPECT_EQ(config->moe_config.shared_expert_inter_size, 1536u);
}

// Optionally validate against the real checkpoint's config.json when present.
TEST_F(HyV3ConfigParserTest, ParsesRealConfigIfPresent) {
  // Prefer the FP8 checkpoint; fall back to the BF16 one.
  const std::vector<std::string> candidates = {"/data1/models/Hy3-FP8-29182", "/data1/models/Hy3-29183"};
  std::string model_dir;
  std::string config_file;
  for (const auto& d : candidates) {
    if (std::filesystem::exists(d + "/config.json")) {
      model_dir = d;
      config_file = d + "/config.json";
      break;
    }
  }
  if (config_file.empty()) {
    GTEST_SKIP() << "Real hy_v3 config not found in any of the candidate paths";
  }
  nlohmann::json config_json = ReadJsonFromFile(config_file);
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, model_dir, base_config).OK());
  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->head_num, 64u);
  EXPECT_EQ(config->moe_config.num_experts, 192u);
  EXPECT_EQ(config->moe_config.first_k_dense_replace, 1u);
  EXPECT_GT(config->rope_theta, 1.0f);
}

// BF16 checkpoint: no quantization_config -> is_quant=false.
TEST_F(HyV3ConfigParserTest, BF16NotQuantized) {
  nlohmann::json config_json = MakeHyV3Config();
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3", base_config).OK());
  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_FALSE(config->is_quant);
}

// FP8-E4M3 static per-tensor: quantization_config with quant_method="fp8".
TEST_F(HyV3ConfigParserTest, ParsesFp8StaticQuantConfig) {
  nlohmann::json config_json = MakeHyV3Config();
  config_json["quantization_config"] = nlohmann::json{
      {"activation_scheme", "static"},
      {"ignored_layers", {"lm_head", "model.embed_tokens"}},
      {"quant_method", "fp8"},
      {"kv_cache_scheme", "static"},
  };
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3-fp8", base_config).OK());
  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_TRUE(config->is_quant);
  EXPECT_EQ(config->quant_config.method, QUANT_FP8_E4M3);
  EXPECT_TRUE(config->quant_config.is_checkpoint_fp8_serialized);
  EXPECT_TRUE(config->quant_config.is_activation_scheme_static);
  EXPECT_FALSE(config->quant_config.is_fp8_blockwise);
  // MTP head is honored (num_nextn_predict_layers=1 in the fixture).
  EXPECT_EQ(config->num_nextn_predict_layers, 1u);
}

// Unknown quant_method must fail loudly instead of silently loading garbage.
TEST_F(HyV3ConfigParserTest, RejectsUnsupportedQuantMethod) {
  nlohmann::json config_json = MakeHyV3Config();
  config_json["quantization_config"] = nlohmann::json{
      {"quant_method", "awq"},
      {"bits", 4},
      {"group_size", 128},
  };
  std::shared_ptr<BaseModelConfig> base_config;
  auto status = parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3-bad", base_config);
  EXPECT_FALSE(status.OK());
}

// rope_theta at top level (no rope_parameters object) should still be read.
TEST_F(HyV3ConfigParserTest, ParsesTopLevelRopeTheta) {
  nlohmann::json config_json = MakeHyV3Config();
  config_json.erase("rope_parameters");
  config_json["rope_theta"] = 500000.0f;
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3", base_config).OK());
  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_FLOAT_EQ(config->rope_theta, 500000.0f);
}

// When num_nextn_predict_layers is absent, it defaults to 0 (no MTP).
TEST_F(HyV3ConfigParserTest, ParsesNoMtp) {
  nlohmann::json config_json = MakeHyV3Config();
  config_json.erase("num_nextn_predict_layers");
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3", base_config).OK());
  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->num_nextn_predict_layers, 0u);
}

// Minimal config: only mandatory fields, everything else uses defaults.
TEST_F(HyV3ConfigParserTest, ParsesMinimalConfigWithDefaults) {
  nlohmann::json config_json = {
      {"architectures", {"HYV3ForCausalLM"}},
      {"model_type", "hy_v3"},
      {"num_experts", 192},
  };
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3-min", base_config).OK());
  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->head_num, 64u);
  EXPECT_EQ(config->num_key_value_heads, 8u);
  EXPECT_EQ(config->hidden_units, 4096u);
  EXPECT_EQ(config->num_layer, 80u);
  EXPECT_EQ(config->moe_config.num_experts, 192u);
  EXPECT_EQ(config->moe_config.experts_topk, 8u);
  EXPECT_FLOAT_EQ(config->rope_theta, 11158840.0f);
  EXPECT_EQ(config->weight_data_type, DataType::TYPE_BF16);
  EXPECT_FALSE(config->is_quant);
}

// softmax routing (moe_router_use_sigmoid=false).
TEST_F(HyV3ConfigParserTest, ParsesSoftmaxRouting) {
  nlohmann::json config_json = MakeHyV3Config();
  config_json["moe_router_use_sigmoid"] = false;
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3", base_config).OK());
  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->moe_config.scoring_func, "softmax");
  // Same caveat as the runtime path (ParseHyV3SoftmaxRouting): topk_method stays "fast", which
  // InvokeMoeCutlassGemm only pairs with sigmoid, so softmax is BF16-only in practice.
  EXPECT_EQ(config->moe_config.topk_method, "fast");
}

// No shared experts (num_shared_experts=0).
TEST_F(HyV3ConfigParserTest, ParsesNoSharedExperts) {
  nlohmann::json config_json = MakeHyV3Config();
  config_json["num_shared_experts"] = 0;
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3", base_config).OK());
  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_FALSE(config->has_shared_experts);
  EXPECT_EQ(config->moe_config.num_shared_experts, 0u);
}

// expert_hidden_dim fallback when moe_intermediate_size is absent.
TEST_F(HyV3ConfigParserTest, ParsesExpertHiddenDimFallback) {
  nlohmann::json config_json = MakeHyV3Config();
  config_json.erase("moe_intermediate_size");
  config_json["expert_hidden_dim"] = 2048;
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3", base_config).OK());
  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->moe_config.moe_inter_size, 2048u);
}

// tie_word_embeddings=true should be respected.
TEST_F(HyV3ConfigParserTest, ParsesTieEmbeddingsTrue) {
  nlohmann::json config_json = MakeHyV3Config();
  config_json["tie_word_embeddings"] = true;
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3", base_config).OK());
  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_TRUE(config->tie_word_embeddings);
}

// Missing tie_word_embeddings key sets exist_tie_embeddings_param=false.
TEST_F(HyV3ConfigParserTest, ParsesMissingTieEmbeddings) {
  nlohmann::json config_json = MakeHyV3Config();
  config_json.erase("tie_word_embeddings");
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3", base_config).OK());
  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_FALSE(config->exist_tie_embeddings_param);
  EXPECT_FALSE(config->tie_word_embeddings);
}

// Non-empty yaml_weight_quant_method should be rejected by ParseQuantConfig.
// Note: ParseQuantConfig is called from ParseModelConfig so we need to inject via Environment.
// Since Singleton<Environment> is involved, we test this indirectly via the already-existing
// RejectsUnsupportedQuantMethod test. Here we simply confirm the explicit fp8 blockwise detection:
TEST_F(HyV3ConfigParserTest, ParsesFp8BlockwiseConfig) {
  nlohmann::json config_json = MakeHyV3Config();
  config_json["quantization_config"] = nlohmann::json{
      {"quant_method", "fp8"},
      {"activation_scheme", "dynamic"},
      {"weight_block_size", {128, 128}},
  };
  std::shared_ptr<BaseModelConfig> base_config;
  ASSERT_TRUE(parser_.ParseModelConfig(config_json, parallel_config_, "/tmp/hy3-blk", base_config).OK());
  auto config = std::dynamic_pointer_cast<HyV3Config>(base_config);
  ASSERT_NE(config, nullptr);
  EXPECT_TRUE(config->is_quant);
  EXPECT_EQ(config->quant_config.method, QUANT_BLOCK_FP8_E4M3);
  EXPECT_TRUE(config->quant_config.is_fp8_blockwise);
}

// =============================================================================
// HyV3MtpDraftModel tests — exercise error paths and boundary logic without GPU.
// Same pattern as mtp_decoder_adapter_test.cpp: pass nullptr for heavy deps that
// the tested code paths never touch.
// =============================================================================

class HyV3MtpDraftModelTest : public testing::Test {
 protected:
  void SetUp() override { InitLoguru(); }
};

TEST_F(HyV3MtpDraftModelTest, ForwardLayerFailsWhenLayersEmpty) {
  std::unordered_map<int, std::shared_ptr<HyV3MtpLayer>> empty_layers;
  HyV3MtpDraftModel model(&empty_layers, nullptr, nullptr, 80, 80);

  std::vector<Tensor> residual_buffer;
  ForwardingContext dummy_ctx;
  auto status = model.ForwardLayer(residual_buffer, dummy_ctx);
  EXPECT_FALSE(status.OK());
  EXPECT_NE(status.GetMessage().find("No MTP layers available"), std::string::npos);
}

TEST_F(HyV3MtpDraftModelTest, ForwardLayerFailsWhenLayerIndexNotFound) {
  std::unordered_map<int, std::shared_ptr<HyV3MtpLayer>> layers;
  layers[80] = nullptr;
  HyV3MtpDraftModel model(&layers, nullptr, nullptr, 81, 81);

  std::vector<Tensor> residual_buffer;
  ForwardingContext dummy_ctx;
  auto status = model.ForwardLayer(residual_buffer, dummy_ctx);
  EXPECT_FALSE(status.OK());
  EXPECT_NE(status.GetMessage().find("not found"), std::string::npos);
}

TEST_F(HyV3MtpDraftModelTest, ForwardMhcNormLmHeadNoopWhenNullPtrs) {
  std::unordered_map<int, std::shared_ptr<HyV3MtpLayer>> empty_layers;
  HyV3MtpDraftModel model(&empty_layers, nullptr, nullptr, 80, 80);

  std::vector<Tensor> residual_buffer;
  std::shared_ptr<BaseWeight> base_weight;
  ForwardingContext dummy_ctx;
  auto status = model.ForwardMhcNormLmHead(residual_buffer, base_weight, dummy_ctx);
  EXPECT_TRUE(status.OK());
}

// =============================================================================
// HyV3 model architecture description — no GPU / no weights needed.
// =============================================================================

TEST(HyV3ModelTest, GetModelRunConfigDescribesRopePreNorm) {
  HyV3 hy_v3;
  ModelRunConfig model_run_config;
  ModelConfig model_config;
  ASSERT_TRUE(hy_v3.GetModelRunConfig(model_run_config, model_config).OK());
  EXPECT_EQ(model_run_config.position_encoding, PositionEncoding::ROPE);
  EXPECT_EQ(model_run_config.layernorm_position, LayerNormPosition::PRE_NORM);
}

// =============================================================================
// Model architecture / factory wiring for hy_v3.
// =============================================================================

TEST(HyV3FactoryTest, ArchitectureStringRoundTrip) {
  ModelArchitecture model_arch = ModelArchitecture::ARCH_UNKNOWN;
  ASSERT_TRUE(GetModelArchitectureFromString("hy_v3", model_arch).OK());
  EXPECT_EQ(model_arch, ModelArchitecture::ARCH_HY_V3);

  std::string model_type;
  ASSERT_TRUE(GetModelTypeFromArchitecture(ModelArchitecture::ARCH_HY_V3, model_type).OK());
  EXPECT_EQ(model_type, "hy_v3");
}

TEST(HyV3FactoryTest, CreateModelConfigParserReturnsHyV3Parser) {
  std::shared_ptr<BaseModelConfigParser> config_parser;
  ASSERT_TRUE(ModelConfigParserFactory::CreateModelConfigParser(ModelArchitecture::ARCH_HY_V3, config_parser).OK());
  ASSERT_NE(config_parser, nullptr);
  EXPECT_NE(std::dynamic_pointer_cast<HyV3ConfigParser>(config_parser), nullptr);
}

// =============================================================================
// HyV3WeightLoader tests — exercise name filtering, method registration and the
// host->device weight routing with synthetic tensors. No checkpoint is read:
// every tensor is fabricated in host memory with the shapes the real hy_v3
// checkpoint ships, so the loader's shape arithmetic is still validated.
// Same fixture shape as deepseek_v4_weight_loader_test / fp8_e4m3_method_test.
// =============================================================================

#ifdef ENABLE_CUDA

class HyV3WeightLoaderTest : public testing::Test {
 protected:
  // Small but shape-consistent stand-in for a real hy_v3 checkpoint:
  // layer 0 is the dense prefix (first_k_dense_replace=1), layer 1 is MoE and
  // layer 2 (== num_layer) is the MTP/nextn layer.
  static constexpr size_t kHeadNum = 8;
  static constexpr size_t kKvHeads = 2;
  static constexpr size_t kHeadDim = 64;
  static constexpr size_t kHiddenUnits = 512;
  static constexpr size_t kInterSize = 1024;
  static constexpr size_t kMoeInterSize = 256;
  static constexpr size_t kNumExperts = 4;
  static constexpr size_t kVocabSize = 1024;
  static constexpr size_t kNumLayer = 2;
  static constexpr size_t kFirstKDenseReplace = 1;
  static constexpr int kMtpLayerIdx = static_cast<int>(kNumLayer);
  static constexpr int kDevRank = 0;

  static constexpr size_t kQSize = kHeadNum * kHeadDim;
  static constexpr size_t kKvSize = kKvHeads * kHeadDim;

  void SetUp() override {
    InitLoguru();

    env_ = Singleton<Environment>::GetInstance();
    context_ = std::make_shared<Context>();
    context_->global.reset(new GlobalContext(1, 1, 1, env_));
    context_->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context_->device = std::make_shared<DeviceContext>(1);
    context_->device->ext->InitNcclParam(nccl_uid);

    env_->SetTensorParallelSize(1);
    // Main layers only by default; the MTP tests widen this.
    SetLayerRange(0, static_cast<int>(kNumLayer) - 1, -1, -1);

    config_ = MakeConfig();
  }

  void TearDown() override { DeviceSynchronize(); }

  // The loader snapshots the pipeline config in its constructor, so this must be
  // called before the loader under test is created.
  void SetLayerRange(int lower_layer_idx, int upper_layer_idx, int lower_nextn_layer_idx, int upper_nextn_layer_idx) {
    PipelineConfig pipeline_config;
    env_->GetPipelineConfig(pipeline_config);
    pipeline_config.lower_layer_idx = lower_layer_idx;
    pipeline_config.upper_layer_idx = upper_layer_idx;
    pipeline_config.lower_nextn_layer_idx = lower_nextn_layer_idx;
    pipeline_config.upper_nextn_layer_idx = upper_nextn_layer_idx;
    env_->SetPipelineConfig(pipeline_config);
  }

  static std::shared_ptr<HyV3Config> MakeConfig() {
    auto config = std::make_shared<HyV3Config>();
    config->model_dir = "/tmp/hy3-fake";
    config->weight_data_type = DataType::TYPE_BF16;
    config->head_num = kHeadNum;
    config->num_key_value_heads = kKvHeads;
    config->size_per_head = kHeadDim;
    config->rotary_embedding = kHeadDim;
    config->hidden_units = kHiddenUnits;
    config->inter_size = kInterSize;
    config->vocab_size = kVocabSize;
    config->num_layer = kNumLayer;
    config->num_nextn_predict_layers = 1;
    config->is_moe = true;
    config->has_shared_experts = true;
    config->use_qk_norm = true;
    config->enable_add_qkv_bias = false;
    config->moe_config.num_experts = kNumExperts;
    config->moe_config.experts_topk = 2;
    config->moe_config.moe_inter_size = kMoeInterSize;
    config->moe_config.first_k_dense_replace = kFirstKDenseReplace;
    config->moe_config.num_shared_experts = 1;
    config->moe_config.shared_expert_inter_size = kMoeInterSize;
    config->moe_config.scoring_func = "sigmoid";
    config->moe_config.use_e_score_correction_bias = true;
    // hy_v3 always stores routed experts in vLLM [gate, up] order.
    config->moe_config.use_vllm_moe = true;
    return config;
  }

  static Tensor MakeHostTensor(const std::vector<size_t>& shape, DataType dtype, float value) {
    Tensor tensor(MemoryLocation::LOCATION_HOST, dtype, shape);
    const size_t element_num = tensor.GetElementNumber();
    if (dtype == DataType::TYPE_FP32) {
      float* ptr = tensor.GetPtr<float>();
      for (size_t i = 0; i < element_num; ++i) {
        ptr[i] = value;
      }
    } else if (dtype == DataType::TYPE_BF16) {
      bfloat16* ptr = tensor.GetPtr<bfloat16>();
      for (size_t i = 0; i < element_num; ++i) {
        ptr[i] = bfloat16(value);
      }
    } else {
      // FP8-E4M3 (and any other narrow type): the loader only memcpys/rescales
      // these, so a byte pattern is enough.
      std::memset(tensor.GetPtr<void>(), 0x3c, tensor.GetTotalBytes());
    }
    return tensor;
  }

  // Every host tensor a single decoder-shaped layer ships. MTP (nextn) layers are
  // decoder-shaped too, so this is reused for them.
  static void AddDecoderLayerWeights(std::unordered_map<std::string, Tensor>& host_weights, int layer_idx) {
    const std::string lp = fmt::format("model.layers.{}", layer_idx);

    host_weights[lp + ".self_attn.q_proj.weight"] = MakeHostTensor({kQSize, kHiddenUnits}, TYPE_BF16, 0.1f);
    host_weights[lp + ".self_attn.k_proj.weight"] = MakeHostTensor({kKvSize, kHiddenUnits}, TYPE_BF16, 0.1f);
    host_weights[lp + ".self_attn.v_proj.weight"] = MakeHostTensor({kKvSize, kHiddenUnits}, TYPE_BF16, 0.1f);
    host_weights[lp + ".self_attn.o_proj.weight"] = MakeHostTensor({kHiddenUnits, kQSize}, TYPE_BF16, 0.1f);

    // Pre-RoPE QK-Norm (renamed from q_norm/k_norm by the weight map).
    host_weights[lp + ".self_attn.query_layernorm.weight"] = MakeHostTensor({kHeadDim}, TYPE_BF16, 1.0f);
    host_weights[lp + ".self_attn.key_layernorm.weight"] = MakeHostTensor({kHeadDim}, TYPE_BF16, 1.0f);

    host_weights[lp + ".input_layernorm.weight"] = MakeHostTensor({kHiddenUnits}, TYPE_BF16, 1.0f);
    host_weights[lp + ".post_attention_layernorm.weight"] = MakeHostTensor({kHiddenUnits}, TYPE_BF16, 1.0f);

    if (layer_idx >= static_cast<int>(kFirstKDenseReplace)) {
      // Router gate + aux-loss-free expert bias (bias is FP32 in the checkpoint).
      host_weights[lp + ".mlp.gate.weight"] = MakeHostTensor({kNumExperts, kHiddenUnits}, TYPE_BF16, 0.1f);
      host_weights[lp + ".mlp.gate.e_score_correction_bias"] = MakeHostTensor({kNumExperts}, TYPE_FP32, 0.1f);

      for (size_t expert_idx = 0; expert_idx < kNumExperts; ++expert_idx) {
        const std::string ep = fmt::format("{}.mlp.experts.{}", lp, expert_idx);
        host_weights[ep + ".gate_proj.weight"] = MakeHostTensor({kMoeInterSize, kHiddenUnits}, TYPE_BF16, 0.1f);
        host_weights[ep + ".up_proj.weight"] = MakeHostTensor({kMoeInterSize, kHiddenUnits}, TYPE_BF16, 0.1f);
        host_weights[ep + ".down_proj.weight"] = MakeHostTensor({kHiddenUnits, kMoeInterSize}, TYPE_BF16, 0.1f);
      }

      host_weights[lp + ".mlp.shared_expert.gate_proj.weight"] =
          MakeHostTensor({kMoeInterSize, kHiddenUnits}, TYPE_BF16, 0.1f);
      host_weights[lp + ".mlp.shared_expert.up_proj.weight"] =
          MakeHostTensor({kMoeInterSize, kHiddenUnits}, TYPE_BF16, 0.1f);
      host_weights[lp + ".mlp.shared_expert.down_proj.weight"] =
          MakeHostTensor({kHiddenUnits, kMoeInterSize}, TYPE_BF16, 0.1f);
    } else {
      // Dense prefix layer.
      host_weights[lp + ".mlp.gate_proj.weight"] = MakeHostTensor({kInterSize, kHiddenUnits}, TYPE_BF16, 0.1f);
      host_weights[lp + ".mlp.up_proj.weight"] = MakeHostTensor({kInterSize, kHiddenUnits}, TYPE_BF16, 0.1f);
      host_weights[lp + ".mlp.down_proj.weight"] = MakeHostTensor({kHiddenUnits, kInterSize}, TYPE_BF16, 0.1f);
    }
  }

  // MTP-only tensors on top of the decoder-shaped ones.
  static void AddMtpExtraWeights(std::unordered_map<std::string, Tensor>& host_weights, int layer_idx) {
    const std::string lp = fmt::format("model.layers.{}", layer_idx);
    host_weights[lp + ".enorm.weight"] = MakeHostTensor({kHiddenUnits}, TYPE_BF16, 1.0f);
    host_weights[lp + ".hnorm.weight"] = MakeHostTensor({kHiddenUnits}, TYPE_BF16, 1.0f);
    host_weights[lp + ".final_layernorm.weight"] = MakeHostTensor({kHiddenUnits}, TYPE_BF16, 1.0f);
    host_weights[lp + ".eh_proj.weight"] = MakeHostTensor({kHiddenUnits, 2 * kHiddenUnits}, TYPE_BF16, 0.1f);
  }

  static void AddBaseWeights(std::unordered_map<std::string, Tensor>& host_weights) {
    host_weights["model.norm.weight"] = MakeHostTensor({kHiddenUnits}, TYPE_BF16, 1.0f);
    host_weights["model.embed_tokens.weight"] = MakeHostTensor({kVocabSize, kHiddenUnits}, TYPE_BF16, 0.1f);
    host_weights["lm_head.weight"] = MakeHostTensor({kVocabSize, kHiddenUnits}, TYPE_BF16, 0.1f);
  }

 protected:
  std::shared_ptr<Environment> env_{nullptr};
  std::shared_ptr<Context> context_{nullptr};
  std::shared_ptr<HyV3Config> config_{nullptr};
};

// The factory must hand back an HyV3WeightLoader for ARCH_HY_V3.
TEST_F(HyV3WeightLoaderTest, FactoryCreatesHyV3WeightLoader) {
  std::shared_ptr<BaseModelWeightLoader> weight_loader;
  ASSERT_TRUE(ModelWeightLoaderFactory::CreateModelWeightLoader(ModelArchitecture::ARCH_HY_V3, config_, env_, context_,
                                                                weight_loader)
                  .OK());
  ASSERT_NE(weight_loader, nullptr);
  EXPECT_NE(std::dynamic_pointer_cast<HyV3WeightLoader>(weight_loader), nullptr);
}

// shared_head.* is shipped by MTP checkpoints but never referenced (the main
// lm_head is reused), so it must be dropped.
TEST_F(HyV3WeightLoaderTest, FilterWeightNamesDropsSharedHead) {
  HyV3WeightLoader weight_loader(config_, env_, context_);
  std::vector<std::string> weight_names = {
      "model.layers.0.self_attn.q_proj.weight", "model.layers.1.mlp.gate.weight", "shared_head.head.weight",
      "shared_head.norm.weight",                "model.embed_tokens.weight",      "lm_head.weight",
  };
  ASSERT_TRUE(weight_loader.FilterWeightNames(weight_names).OK());

  const std::set<std::string> remaining(weight_names.begin(), weight_names.end());
  EXPECT_FALSE(remaining.count("shared_head.head.weight"));
  EXPECT_FALSE(remaining.count("shared_head.norm.weight"));
  // Everything else survives in standalone mode.
  EXPECT_TRUE(remaining.count("model.layers.0.self_attn.q_proj.weight"));
  EXPECT_TRUE(remaining.count("model.layers.1.mlp.gate.weight"));
  EXPECT_TRUE(remaining.count("model.embed_tokens.weight"));
  EXPECT_TRUE(remaining.count("lm_head.weight"));
}

// MTP is off (lower/upper_nextn_layer_idx < 0): tensors at layer >= num_layer
// must be dropped instead of being loaded for a layer that is never built.
TEST_F(HyV3WeightLoaderTest, FilterWeightNamesDropsMtpLayersWhenMtpDisabled) {
  SetLayerRange(0, static_cast<int>(kNumLayer) - 1, -1, -1);
  HyV3WeightLoader weight_loader(config_, env_, context_);

  std::vector<std::string> weight_names = {
      "model.layers.0.self_attn.q_proj.weight",
      "model.layers.1.self_attn.q_proj.weight",
      "model.layers.2.self_attn.q_proj.weight",  // MTP layer, MTP disabled
      "model.layers.2.eh_proj.weight",           // MTP layer, MTP disabled
      "model.embed_tokens.weight",
  };
  ASSERT_TRUE(weight_loader.FilterWeightNames(weight_names).OK());

  const std::set<std::string> remaining(weight_names.begin(), weight_names.end());
  EXPECT_TRUE(remaining.count("model.layers.0.self_attn.q_proj.weight"));
  EXPECT_TRUE(remaining.count("model.layers.1.self_attn.q_proj.weight"));
  EXPECT_FALSE(remaining.count("model.layers.2.self_attn.q_proj.weight"));
  EXPECT_FALSE(remaining.count("model.layers.2.eh_proj.weight"));
  EXPECT_TRUE(remaining.count("model.embed_tokens.weight"));
}

// MTP active: the nextn layer is kept, anything past it is still dropped.
TEST_F(HyV3WeightLoaderTest, FilterWeightNamesKeepsMtpLayersWhenMtpEnabled) {
  SetLayerRange(0, static_cast<int>(kNumLayer) - 1, kMtpLayerIdx, kMtpLayerIdx);
  HyV3WeightLoader weight_loader(config_, env_, context_);

  std::vector<std::string> weight_names = {
      "model.layers.1.self_attn.q_proj.weight",
      "model.layers.2.self_attn.q_proj.weight",  // within nextn range
      "model.layers.2.final_layernorm.weight",   // loaded as nextn_shared_head_norm_
      "model.layers.3.self_attn.q_proj.weight",  // beyond nextn range
  };
  ASSERT_TRUE(weight_loader.FilterWeightNames(weight_names).OK());

  const std::set<std::string> remaining(weight_names.begin(), weight_names.end());
  EXPECT_TRUE(remaining.count("model.layers.1.self_attn.q_proj.weight"));
  EXPECT_TRUE(remaining.count("model.layers.2.self_attn.q_proj.weight"));
  EXPECT_TRUE(remaining.count("model.layers.2.final_layernorm.weight"));
  EXPECT_FALSE(remaining.count("model.layers.3.self_attn.q_proj.weight"));
}

// Pipeline-parallel worker: layers outside [lower, upper] belong to other ranks.
TEST_F(HyV3WeightLoaderTest, FilterWeightNamesDropsLayersOutsidePipelineRange) {
  auto config = MakeConfig();
  config->num_layer = 4;
  SetLayerRange(1, 2, -1, -1);
  HyV3WeightLoader weight_loader(config, env_, context_);

  std::vector<std::string> weight_names = {
      "model.layers.0.self_attn.q_proj.weight", "model.layers.1.self_attn.q_proj.weight",
      "model.layers.2.self_attn.q_proj.weight", "model.layers.3.self_attn.q_proj.weight",
      "model.norm.weight",
  };
  ASSERT_TRUE(weight_loader.FilterWeightNames(weight_names).OK());

  const std::set<std::string> remaining(weight_names.begin(), weight_names.end());
  EXPECT_FALSE(remaining.count("model.layers.0.self_attn.q_proj.weight"));
  EXPECT_TRUE(remaining.count("model.layers.1.self_attn.q_proj.weight"));
  EXPECT_TRUE(remaining.count("model.layers.2.self_attn.q_proj.weight"));
  EXPECT_FALSE(remaining.count("model.layers.3.self_attn.q_proj.weight"));
  // Non-layer tensors (layer_idx == -1) are untouched by the range check.
  EXPECT_TRUE(remaining.count("model.norm.weight"));
}

// BF16 checkpoint registers the common (non-quant) load/process methods.
TEST_F(HyV3WeightLoaderTest, PreProcessRegistersCommonMethodsForBf16) {
  HyV3WeightLoader weight_loader(config_, env_, context_);
  std::unordered_map<std::string, Tensor> host_weights;
  EXPECT_TRUE(weight_loader.PreProcessModelWeights(host_weights).OK());
}

// FP8 checkpoints (static per-tensor and blockwise) take the Fp8E4m3Method branch.
TEST_F(HyV3WeightLoaderTest, PreProcessRegistersFp8MethodsForQuantizedCheckpoints) {
  std::unordered_map<std::string, Tensor> host_weights;

  {
    auto config = MakeConfig();
    config->is_quant = true;
    config->quant_config.method = QUANT_FP8_E4M3;
    config->quant_config.is_checkpoint_fp8_serialized = true;
    config->quant_config.is_activation_scheme_static = true;
    HyV3WeightLoader weight_loader(config, env_, context_);
    EXPECT_TRUE(weight_loader.PreProcessModelWeights(host_weights).OK());
  }

  {
    auto config = MakeConfig();
    config->is_quant = true;
    config->quant_config.method = QUANT_BLOCK_FP8_E4M3;
    config->quant_config.is_fp8_blockwise = true;
    config->quant_config.weight_block_size = {128, 128};
    HyV3WeightLoader weight_loader(config, env_, context_);
    EXPECT_TRUE(weight_loader.PreProcessModelWeights(host_weights).OK());
  }
}

// Every weight family hy_v3 ships must be routed to a handler by
// ProcessModelWeights: attention, QK-Norm, router gate + expert bias, routed
// experts, shared expert, dense prefix MLP, norms, MTP eh_proj, embed, lm_head.
TEST_F(HyV3WeightLoaderTest, ProcessModelWeightsRoutesEveryWeightFamily) {
  SetLayerRange(0, static_cast<int>(kNumLayer) - 1, kMtpLayerIdx, kMtpLayerIdx);
  HyV3WeightLoader weight_loader(config_, env_, context_);

  std::unordered_map<std::string, Tensor> host_weights;
  ASSERT_TRUE(weight_loader.PreProcessModelWeights(host_weights).OK());

  AddDecoderLayerWeights(host_weights, 0);  // dense prefix
  AddDecoderLayerWeights(host_weights, 1);  // MoE
  AddDecoderLayerWeights(host_weights, kMtpLayerIdx);
  AddMtpExtraWeights(host_weights, kMtpLayerIdx);
  AddBaseWeights(host_weights);

  std::unordered_map<std::string, Tensor> device_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;
  ASSERT_TRUE(weight_loader.ProcessModelWeights(host_weights, kDevRank, device_weights, left_host_weights).OK());

  // Attention + QK-Norm.
  EXPECT_TRUE(device_weights.count("model.layers.1.self_attn.q_proj.weight"));
  EXPECT_TRUE(device_weights.count("model.layers.1.self_attn.o_proj.weight"));
  EXPECT_TRUE(device_weights.count("model.layers.1.self_attn.query_layernorm.weight"));
  EXPECT_TRUE(device_weights.count("model.layers.1.self_attn.key_layernorm.weight"));

  // Router gate keeps its raw layout here (transposed in PostProcess); the
  // aux-loss-free bias is moved to device as-is, in FP32.
  EXPECT_TRUE(device_weights.count("model.layers.1.mlp.gate.weight"));
  ASSERT_TRUE(device_weights.count("model.layers.1.mlp.gate.e_score_correction_bias"));
  EXPECT_EQ(device_weights.at("model.layers.1.mlp.gate.e_score_correction_bias").dtype, DataType::TYPE_FP32);

  // Routed experts and the single shared expert.
  EXPECT_TRUE(device_weights.count("model.layers.1.mlp.experts.0.gate_proj.weight"));
  EXPECT_TRUE(device_weights.count("model.layers.1.mlp.experts.0.up_proj.weight"));
  EXPECT_TRUE(device_weights.count("model.layers.1.mlp.experts.0.down_proj.weight"));
  EXPECT_TRUE(device_weights.count("model.layers.1.mlp.shared_expert.gate_proj.weight"));
  EXPECT_TRUE(device_weights.count("model.layers.1.mlp.shared_expert.down_proj.weight"));

  // Dense prefix layer.
  EXPECT_TRUE(device_weights.count("model.layers.0.mlp.gate_proj.weight"));
  EXPECT_TRUE(device_weights.count("model.layers.0.mlp.down_proj.weight"));

  // Norms, including the MTP-only ones.
  EXPECT_TRUE(device_weights.count("model.norm.weight"));
  EXPECT_TRUE(device_weights.count("model.layers.1.input_layernorm.weight"));
  EXPECT_TRUE(device_weights.count("model.layers.1.post_attention_layernorm.weight"));
  EXPECT_TRUE(device_weights.count(fmt::format("model.layers.{}.enorm.weight", kMtpLayerIdx)));
  EXPECT_TRUE(device_weights.count(fmt::format("model.layers.{}.hnorm.weight", kMtpLayerIdx)));
  EXPECT_TRUE(device_weights.count(fmt::format("model.layers.{}.final_layernorm.weight", kMtpLayerIdx)));

  // eh_proj is split on axis 0 then transposed, so [hidden, 2*hidden] -> [2*hidden, hidden] at tp=1.
  const std::string eh_proj_name = fmt::format("model.layers.{}.eh_proj.weight", kMtpLayerIdx);
  ASSERT_TRUE(device_weights.count(eh_proj_name));
  ASSERT_EQ(device_weights.at(eh_proj_name).shape.size(), 2u);
  EXPECT_EQ(device_weights.at(eh_proj_name).shape[0], 2 * kHiddenUnits);
  EXPECT_EQ(device_weights.at(eh_proj_name).shape[1], kHiddenUnits);

  EXPECT_TRUE(device_weights.count("model.embed_tokens.weight"));
  EXPECT_TRUE(device_weights.count("lm_head.weight"));
}

// PostProcess fuses QKV, permutes o_proj / down / lm_head and batch-merges the
// routed experts into [num_experts, ...] tensors — for main and MTP layers alike.
TEST_F(HyV3WeightLoaderTest, PostProcessModelWeightsFusesMainAndMtpLayers) {
  SetLayerRange(0, static_cast<int>(kNumLayer) - 1, kMtpLayerIdx, kMtpLayerIdx);
  HyV3WeightLoader weight_loader(config_, env_, context_);

  std::unordered_map<std::string, Tensor> host_weights;
  ASSERT_TRUE(weight_loader.PreProcessModelWeights(host_weights).OK());

  AddDecoderLayerWeights(host_weights, 0);
  AddDecoderLayerWeights(host_weights, 1);
  AddDecoderLayerWeights(host_weights, kMtpLayerIdx);
  AddMtpExtraWeights(host_weights, kMtpLayerIdx);
  AddBaseWeights(host_weights);

  std::unordered_map<std::string, Tensor> device_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;
  ASSERT_TRUE(weight_loader.ProcessModelWeights(host_weights, kDevRank, device_weights, left_host_weights).OK());
  ASSERT_TRUE(weight_loader.PostProcessModelWeights(device_weights, kDevRank).OK());

  for (int layer_idx : {0, 1, kMtpLayerIdx}) {
    const std::string lp = fmt::format("model.layers.{}", layer_idx);

    // Q/K/V fused into one column-major tensor: [hidden, q + 2 * kv] after permute.
    const std::string qkv_name = lp + ".self_attn.query_key_value.weight";
    ASSERT_TRUE(device_weights.count(qkv_name)) << qkv_name;
    ASSERT_EQ(device_weights.at(qkv_name).shape.size(), 2u);
    EXPECT_EQ(device_weights.at(qkv_name).shape[0], kHiddenUnits);
    EXPECT_EQ(device_weights.at(qkv_name).shape[1], kQSize + 2 * kKvSize);
    EXPECT_FALSE(device_weights.count(lp + ".self_attn.q_proj.weight")) << "q_proj should be consumed by the fusion";

    if (layer_idx >= static_cast<int>(kFirstKDenseReplace)) {
      // Routed experts merged to [num_experts, 2 * moe_inter / tp, hidden].
      const std::string up_gate_name = lp + ".mlp.experts.up_gate_proj.weight";
      ASSERT_TRUE(device_weights.count(up_gate_name)) << up_gate_name;
      ASSERT_EQ(device_weights.at(up_gate_name).shape.size(), 3u);
      EXPECT_EQ(device_weights.at(up_gate_name).shape[0], kNumExperts);
      EXPECT_EQ(device_weights.at(up_gate_name).shape[1], 2 * kMoeInterSize);
      EXPECT_EQ(device_weights.at(up_gate_name).shape[2], kHiddenUnits);

      // ... and down to [num_experts, hidden, moe_inter / tp].
      const std::string down_name = lp + ".mlp.experts.down_proj.weight";
      ASSERT_TRUE(device_weights.count(down_name)) << down_name;
      ASSERT_EQ(device_weights.at(down_name).shape.size(), 3u);
      EXPECT_EQ(device_weights.at(down_name).shape[0], kNumExperts);
      EXPECT_EQ(device_weights.at(down_name).shape[1], kHiddenUnits);
      EXPECT_EQ(device_weights.at(down_name).shape[2], kMoeInterSize);

      // Per-expert tensors are consumed by the merge.
      EXPECT_FALSE(device_weights.count(lp + ".mlp.experts.0.gate_proj.weight"));

      // Shared expert gate/up fused.
      EXPECT_TRUE(device_weights.count(lp + ".mlp.shared_expert.gate_up_proj.weight"));
      EXPECT_TRUE(device_weights.count(lp + ".mlp.shared_expert.down_proj.weight"));
    } else {
      // Dense prefix layer fuses into gate_up_proj as well.
      const std::string dense_gate_up_name = lp + ".mlp.gate_up_proj.weight";
      ASSERT_TRUE(device_weights.count(dense_gate_up_name)) << dense_gate_up_name;
      ASSERT_EQ(device_weights.at(dense_gate_up_name).shape.size(), 2u);
      EXPECT_EQ(device_weights.at(dense_gate_up_name).shape[0], kHiddenUnits);
      EXPECT_EQ(device_weights.at(dense_gate_up_name).shape[1], 2 * kInterSize);
    }
  }

  // lm_head is permuted to [hidden, vocab].
  ASSERT_TRUE(device_weights.count("lm_head.weight"));
  ASSERT_EQ(device_weights.at("lm_head.weight").shape.size(), 2u);
  EXPECT_EQ(device_weights.at("lm_head.weight").shape[0], kHiddenUnits);
  EXPECT_EQ(device_weights.at("lm_head.weight").shape[1], kVocabSize);
}

// With MTP disabled, PostProcess must not walk the nextn layer range.
TEST_F(HyV3WeightLoaderTest, PostProcessModelWeightsSkipsMtpWhenDisabled) {
  SetLayerRange(0, static_cast<int>(kNumLayer) - 1, -1, -1);
  HyV3WeightLoader weight_loader(config_, env_, context_);

  std::unordered_map<std::string, Tensor> host_weights;
  ASSERT_TRUE(weight_loader.PreProcessModelWeights(host_weights).OK());
  AddDecoderLayerWeights(host_weights, 0);
  AddDecoderLayerWeights(host_weights, 1);
  AddBaseWeights(host_weights);

  std::unordered_map<std::string, Tensor> device_weights;
  std::unordered_map<std::string, Tensor> left_host_weights;
  ASSERT_TRUE(weight_loader.ProcessModelWeights(host_weights, kDevRank, device_weights, left_host_weights).OK());
  ASSERT_TRUE(weight_loader.PostProcessModelWeights(device_weights, kDevRank).OK());

  EXPECT_TRUE(device_weights.count("model.layers.1.mlp.experts.up_gate_proj.weight"));
  EXPECT_FALSE(device_weights.count(fmt::format("model.layers.{}.mlp.experts.up_gate_proj.weight", kMtpLayerIdx)));
}

// =============================================================================
// HyV3Model layer creation — builds the real model object graph (decoder layers,
// MoE blocks, MTP/nextn layers and the MTP draft model) against a fake weight
// provider. No checkpoint is read: the fake weight hands out correctly shaped
// device tensors on demand, which is all the layer factories inspect at
// construction time. Same pattern as dsv4_mqa_test's Dsv4MqaWeight.
// =============================================================================

class HyV3FakeWeight : public BaseWeight {
 public:
  HyV3FakeWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                 std::shared_ptr<Context> context)
      : BaseWeight(model_config, runtime_config, rank, context) {}
  ~HyV3FakeWeight() override = default;

  Tensor GetModelWeights(const std::string& weight_name) override {
    auto it = weights_map_.find(weight_name);
    if (it != weights_map_.end()) {
      return it->second;
    }
    return Tensor();
  }

  // Register every weight the hy_v3 layers ask for, with the post-load shapes
  // HyV3WeightLoader produces (see PostProcessModelWeights).
  void AddHyV3Weights() {
    const size_t hidden_units = model_config_.hidden_units;
    const size_t q_size = model_config_.head_num * model_config_.size_per_head;
    const size_t kv_size = model_config_.num_key_value_heads * model_config_.size_per_head;
    const size_t head_dim = model_config_.size_per_head;
    const size_t inter_size = model_config_.inter_size;
    const size_t moe_inter_size = model_config_.moe_config.moe_inter_size;
    const size_t shared_inter_size = model_config_.moe_config.shared_expert_inter_size;
    const size_t num_experts = model_config_.moe_config.num_experts;
    const size_t vocab_size = model_config_.vocab_size;
    const int first_k_dense_replace = static_cast<int>(model_config_.moe_config.first_k_dense_replace);
    const int num_layer = static_cast<int>(model_config_.num_layer);
    const int num_nextn = static_cast<int>(model_config_.num_nextn_predict_layers);

    Add("model.embed_tokens.weight", {vocab_size, hidden_units});
    Add("model.norm.weight", {hidden_units});
    Add("lm_head.weight", {hidden_units, vocab_size});

    for (int layer_idx = 0; layer_idx < num_layer + num_nextn; ++layer_idx) {
      const std::string lp = fmt::format("model.layers.{}", layer_idx);

      Add(lp + ".input_layernorm.weight", {hidden_units});
      Add(lp + ".post_attention_layernorm.weight", {hidden_units});
      Add(lp + ".self_attn.query_key_value.weight", {hidden_units, q_size + 2 * kv_size});
      Add(lp + ".self_attn.o_proj.weight", {q_size, hidden_units});
      Add(lp + ".self_attn.query_layernorm.weight", {head_dim});
      Add(lp + ".self_attn.key_layernorm.weight", {head_dim});

      if (layer_idx >= first_k_dense_replace) {
        Add(lp + ".mlp.gate.weight", {hidden_units, num_experts});
        Add(lp + ".mlp.gate.e_score_correction_bias", {num_experts}, DataType::TYPE_FP32);
        // Batched expert weights: [num_experts, 2 * moe_inter, hidden] / [num_experts, hidden, moe_inter].
        Add(lp + ".mlp.experts.up_gate_proj.weight", {num_experts, 2 * moe_inter_size, hidden_units});
        Add(lp + ".mlp.experts.down_proj.weight", {num_experts, hidden_units, moe_inter_size});
        Add(lp + ".mlp.shared_expert.gate_up_proj.weight", {hidden_units, 2 * shared_inter_size});
        Add(lp + ".mlp.shared_expert.down_proj.weight", {shared_inter_size, hidden_units});
      } else {
        Add(lp + ".mlp.gate_up_proj.weight", {hidden_units, 2 * inter_size});
        Add(lp + ".mlp.down_proj.weight", {inter_size, hidden_units});
      }

      if (layer_idx >= num_layer) {
        // MTP (nextn) extras.
        Add(lp + ".enorm.weight", {hidden_units});
        Add(lp + ".hnorm.weight", {hidden_units});
        Add(lp + ".final_layernorm.weight", {hidden_units});
        Add(lp + ".eh_proj.weight", {2 * hidden_units, hidden_units});
      }
    }
  }

 private:
  using BaseWeight::model_config_;
  using BaseWeight::weights_map_;

  void Add(const std::string& weight_name, const std::vector<size_t>& shape, DataType dtype = DataType::TYPE_BF16) {
    Tensor tensor(MemoryLocation::LOCATION_DEVICE, dtype, shape);
    Memset(tensor.GetPtr<void>(), 0, tensor.GetTotalBytes());
    weights_map_[weight_name] = tensor;
  }
};

class HyV3ModelCreationTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    AttentionBackendManager::GetInstance()->Initialize();

    // A weights-free model dir is enough: only config.json is read here.
    model_dir_ = (std::filesystem::temp_directory_path() / "hy_v3_model_creation_test").string();
    std::filesystem::create_directories(model_dir_);
    std::ofstream ofs(model_dir_ + "/config.json");
    ofs << MakeSmallHyV3ConfigJson().dump(2);
    ofs.close();

    env_ = Singleton<Environment>::GetInstance();
    context_ = std::make_shared<Context>();
    context_->global.reset(new GlobalContext(1, 1, 1, env_));
    context_->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context_->device = std::make_shared<DeviceContext>(1);
    context_->device->ext->InitNcclParam(nccl_uid);

    const std::filesystem::path yaml_path =
        std::filesystem::path(__FILE__).parent_path() / "../../../examples/ksana_llm_deepseekv2.yaml";
    env_->ParseConfig(std::filesystem::absolute(yaml_path).string(), model_dir_);
    env_->UpdateModelConfig();
    env_->GetModelConfig(model_config_);

    // model_instance.cpp flips these for hy_v3 (pre-RoPE QK-Norm instead of post-RoPE).
    model_config_.enable_qk_pre_norm_before_rotary_pos = true;
    model_config_.use_qk_norm = false;
    env_->SetModelConfig(model_config_);

    // Run one MTP step so the nextn layers are built. UpdateModelConfig copies
    // mtp_step_num into the runtime config.
    BatchSchedulerConfig batch_scheduler_config;
    env_->GetBatchSchedulerConfig(batch_scheduler_config);
    batch_scheduler_config.mtp_step_num = 1;
    batch_scheduler_config.max_batch_size = 4;
    batch_scheduler_config.max_step_token_num = 128;
    batch_scheduler_config.max_token_len = 128;
    env_->SetBatchSchedulerConfig(batch_scheduler_config);

    env_->UpdateModelConfig();
    env_->GetModelConfig(model_config_);

    InitPipelineConfigForTest(env_, model_config_, /* enable_mtp */ true);

    block_cache_layout_ = GetTestModelCacheLayout(env_);
    env_->InitializeBlockManagerConfig(block_cache_layout_);

    BlockManagerConfig block_manager_config;
    env_->GetBlockManagerConfig(block_manager_config);
    block_manager_config.device_allocator_config.blocks_num = 8;
    block_manager_config.host_allocator_config.blocks_num = 8;
    env_->SetBlockManagerConfig(block_manager_config);

    env_->GetRuntimeConfig(runtime_config_);
    env_->GetPipelineConfig(pipeline_config_);
  }

  void TearDown() override {
    DeviceSynchronize();
    std::filesystem::remove_all(model_dir_);
  }

  // Tiny hy_v3: 2 main layers (layer 0 dense, layer 1 MoE) + 1 MTP layer.
  static nlohmann::json MakeSmallHyV3ConfigJson() {
    return nlohmann::json{
        {"architectures", {"HYV3ForCausalLM"}},
        {"model_type", "hy_v3"},
        {"torch_dtype", "bfloat16"},
        {"bos_token_id", 1},
        {"eos_token_id", 2},
        {"pad_token_id", 0},
        {"hidden_size", 512},
        {"intermediate_size", 1024},
        {"head_dim", 64},
        {"num_attention_heads", 8},
        {"num_key_value_heads", 2},
        {"num_hidden_layers", 2},
        {"vocab_size", 1024},
        {"rms_norm_eps", 1e-05},
        {"max_position_embeddings", 4096},
        {"qk_norm", true},
        {"tie_word_embeddings", false},
        {"num_experts", 4},
        {"num_experts_per_tok", 2},
        {"moe_intermediate_size", 256},
        {"num_shared_experts", 1},
        {"first_k_dense_replace", 1},
        {"moe_router_use_sigmoid", true},
        {"moe_router_enable_expert_bias", true},
        {"route_norm", true},
        {"router_scaling_factor", 1.0},
        {"num_nextn_predict_layers", 1},
        {"rope_parameters", {{"rope_theta", 10000.0}, {"rope_type", "default"}}},
    };
  }

 protected:
  std::string model_dir_;
  std::shared_ptr<Environment> env_{nullptr};
  std::shared_ptr<Context> context_{nullptr};
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  PipelineConfig pipeline_config_;
  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_{nullptr};
};

// The config.json must land as an hy_v3 MoE model config with an MTP head.
TEST_F(HyV3ModelCreationTest, ModelConfigIsHyV3Moe) {
  EXPECT_EQ(model_config_.type, "hy_v3");
  EXPECT_TRUE(model_config_.is_moe);
  EXPECT_EQ(model_config_.num_layer, 2u);
  EXPECT_EQ(model_config_.moe_config.num_experts, 4u);
  EXPECT_EQ(model_config_.moe_config.first_k_dense_replace, 1u);
  EXPECT_EQ(model_config_.num_nextn_predict_layers, 1u);
  // The dense prefix layer and the MoE layer are both on this (single) node.
  EXPECT_EQ(pipeline_config_.lower_layer_idx, 0);
  EXPECT_EQ(pipeline_config_.upper_layer_idx, 1);
  // MTP layer sits at index num_layer.
  EXPECT_EQ(pipeline_config_.lower_nextn_layer_idx, 2);
  EXPECT_EQ(pipeline_config_.upper_nextn_layer_idx, 2);
  // One MTP step is configured, which is what drives the nextn hidden buffer.
  EXPECT_EQ(runtime_config_.mtp_step_num, 1u);
}

// Building the model exercises HyV3::CreateLayers (dense prefix + MoE decoder
// layers), HyV3Model::CreateLayers (MTP layers, per-MTP final_layernorm and the
// HyV3MtpDraftModel) without touching a checkpoint.
TEST_F(HyV3ModelCreationTest, CreatesDecoderMoeAndMtpLayers) {
  auto fake_weight = std::make_shared<HyV3FakeWeight>(model_config_, runtime_config_, 0, context_);
  fake_weight->AddHyV3Weights();

  SetDevice(0);
  std::shared_ptr<BaseWeight> base_weight = fake_weight;
  auto model = std::make_shared<HyV3Model>(model_config_, runtime_config_, /* rank */ 0, context_, base_weight,
                                           block_cache_layout_);
  ASSERT_NE(model, nullptr);
  // The forwarding context is fully wired up (buffers + attention meta creator).
  ASSERT_NE(model->GetForwardingContext(), nullptr);

  // Proof that HyV3Model::CreateLayers took the MTP branch: it is the only place
  // that installs an HyV3MtpDraftModel and injects the next-step hidden buffer.
  auto* draft_model = model->GetDraftModel();
  ASSERT_NE(draft_model, nullptr) << "MTP draft model should be created when nextn layers are on this node";
  EXPECT_NE(dynamic_cast<HyV3MtpDraftModel*>(draft_model), nullptr);
  EXPECT_TRUE(model->GetForwardingContext()->GetForwardingBuffers()->HasExternalNextStepHiddenBuffer());

  StreamSynchronize(context_->device->GetMemoryManageStream());

  model.reset();
  DeviceSynchronize();
}

#endif  // ENABLE_CUDA

}  // namespace
