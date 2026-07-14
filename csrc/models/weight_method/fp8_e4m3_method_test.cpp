/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/weight_method/fp8_e4m3_method.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "helpers/model_test_helper.h"
#include "test.h"

#include "configure/context.h"
#include "configure/environment.h"
#include "models/base/common_model_weight_loader.h"
#include "utils/singleton.h"
#include "utils/tensor.h"

using namespace ksana_llm;

namespace {

// Create QuantConfig for static FP8 (checkpoint serialized with static activation scheme)
static inline QuantConfig CreateStaticFp8QuantConfig() {
  QuantConfig qc;
  qc.method = QUANT_FP8_E4M3;
  qc.is_checkpoint_fp8_serialized = true;
  qc.is_activation_scheme_static = true;
  return qc;
}

// Create QuantConfig for dynamic FP8 (runtime quantization)
static inline QuantConfig CreateDynamicFp8QuantConfig() {
  QuantConfig qc;
  qc.method = QUANT_FP8_E4M3;
  qc.is_checkpoint_fp8_serialized = false;
  qc.is_activation_scheme_static = false;
  return qc;
}

}  // namespace

class Fp8E4m3MethodTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();

    const auto& env = Singleton<Environment>::GetInstance();

    context_ = std::make_shared<Context>();
    context_->global.reset(new GlobalContext(1, 1, 1, env));
    context_->global->InitDistributedConfig();

    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context_->device = std::make_shared<DeviceContext>(1);
    context_->device->ext->InitNcclParam(nccl_uid);

    env_ = env;
    model_config_ = std::make_shared<BaseModelConfig>();
    common_weight_loader_ = std::make_shared<CommonModelWeightLoader>(model_config_, env_, context_);
  }

  void TearDown() override { device_model_weights_.clear(); }

 protected:
  std::shared_ptr<Context> context_{nullptr};
  std::shared_ptr<Environment> env_{nullptr};
  std::shared_ptr<BaseModelConfig> model_config_{nullptr};
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_{nullptr};
  std::unordered_map<std::string, Tensor> device_model_weights_;

  int dev_rank_{0};
  int tp_{2};
};

// Helper function to verify FP8 processed weights have proper scale bindings
static void ExpectFp8ProcessedAndBound(std::unordered_map<std::string, Tensor>& weights, const std::string& prefix,
                                       bool expect_input_scale = true) {
  const std::string weight_scale_name = prefix + "weight_scale";
  const std::string weight_name = prefix + "weight";

  EXPECT_TRUE(weights.find(weight_name) != weights.end()) << "Weight not found: " << weight_name;
  EXPECT_TRUE(weights.find(weight_scale_name) != weights.end()) << "Weight scale not found: " << weight_scale_name;

  // Weight should have weight_scales bound
  EXPECT_TRUE(weights[weight_name].weight_scales != nullptr) << "Weight scales not bound for: " << weight_name;

  if (expect_input_scale) {
    const std::string input_scale_name = prefix + "input_scale";
    EXPECT_TRUE(weights.find(input_scale_name) != weights.end()) << "Input scale not found: " << input_scale_name;
    EXPECT_TRUE(weights[weight_name].input_scales != nullptr) << "Input scales not bound for: " << weight_name;
  }
}

// Test for static FP8: load and process attention Q/K/V projection
TEST_F(Fp8E4m3MethodTest, TestLoadAndProcessAttnQKVProj_StaticFp8) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateStaticFp8QuantConfig();
  auto fp8_method = std::make_shared<Fp8E4m3Method>(common_weight_loader_, tp_, quant_config);

  const size_t N = 256;  // out features (hidden_size)
  const size_t K = 128;  // in features (hidden_size)

  const std::string q_prefix = "model.layers.0.self_attn.q_proj.";
  const std::string k_prefix = "model.layers.0.self_attn.k_proj.";
  const std::string v_prefix = "model.layers.0.self_attn.v_proj.";

  for (const auto& prefix : {q_prefix, k_prefix, v_prefix}) {
    // Static FP8 weights: FP8 weight tensor with weight_scale and input_scale.
    Tensor weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP8_E4M3, {N, K});
    Tensor weight_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});
    Tensor input_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});

    fp8_method->load_attn_q_k_v_proj(device_model_weights_, prefix + "weight", weight, dev_rank_);
    fp8_method->load_attn_q_k_v_proj(device_model_weights_, prefix + "weight_scale", weight_scale, dev_rank_);
    fp8_method->load_attn_q_k_v_proj(device_model_weights_, prefix + "input_scale", input_scale, dev_rank_);

    EXPECT_TRUE(device_model_weights_.find(prefix + "weight") != device_model_weights_.end());
    EXPECT_TRUE(device_model_weights_.find(prefix + "weight_scale") != device_model_weights_.end());
    EXPECT_TRUE(device_model_weights_.find(prefix + "input_scale") != device_model_weights_.end());

    // RowPara split for qkv_proj => dim=0 split
    EXPECT_EQ(device_model_weights_[prefix + "weight"].shape[0], N / tp_);
  }

  // Process qkv: merge and bind scales
  fp8_method->process_attn_qkv_proj(device_model_weights_, q_prefix, dev_rank_);

  const std::string qkv_prefix = "model.layers.0.self_attn.query_key_value.";
  ExpectFp8ProcessedAndBound(device_model_weights_, qkv_prefix, /*expect_input_scale=*/true);

  // Original q/k/v weight_scale and input_scale should be removed
  EXPECT_TRUE(device_model_weights_.find(q_prefix + "weight_scale") == device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(k_prefix + "weight_scale") == device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(v_prefix + "weight_scale") == device_model_weights_.end());
#endif
}

// Test for dynamic FP8: load and process attention Q/K/V projection
TEST_F(Fp8E4m3MethodTest, TestLoadAndProcessAttnQKVProj_DynamicFp8) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateDynamicFp8QuantConfig();
  auto fp8_method = std::make_shared<Fp8E4m3Method>(common_weight_loader_, tp_, quant_config);

  const size_t N = 256;  // out features
  const size_t K = 128;  // in features

  const std::string q_prefix = "model.layers.0.self_attn.q_proj.";
  const std::string k_prefix = "model.layers.0.self_attn.k_proj.";
  const std::string v_prefix = "model.layers.0.self_attn.v_proj.";

  for (const auto& prefix : {q_prefix, k_prefix, v_prefix}) {
    // Dynamic FP8: load BF16/FP16 weights, will be quantized to FP8 at runtime
    Tensor weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_BF16, {N, K});

    fp8_method->load_attn_q_k_v_proj(device_model_weights_, prefix + "weight", weight, dev_rank_);

    EXPECT_TRUE(device_model_weights_.find(prefix + "weight") != device_model_weights_.end());
    // RowPara split for qkv_proj => dim=0 split
    EXPECT_EQ(device_model_weights_[prefix + "weight"].shape[0], N / tp_);
  }

  // Process qkv: merge and convert to FP8
  fp8_method->process_attn_qkv_proj(device_model_weights_, q_prefix, dev_rank_);

  const std::string qkv_prefix = "model.layers.0.self_attn.query_key_value.";

  // After processing, weight should be FP8 type
  EXPECT_TRUE(device_model_weights_.find(qkv_prefix + "weight") != device_model_weights_.end());
  EXPECT_EQ(device_model_weights_[qkv_prefix + "weight"].dtype, DataType::TYPE_FP8_E4M3);

  // Dynamic FP8: weight_scale should be created and bound, but no input_scale
  EXPECT_TRUE(device_model_weights_.find(qkv_prefix + "weight_scale") != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_[qkv_prefix + "weight"].weight_scales != nullptr);
#endif
}

// Test for static FP8: load and process attention O projection
TEST_F(Fp8E4m3MethodTest, TestLoadAndProcessAttnOProj_StaticFp8) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateStaticFp8QuantConfig();
  auto fp8_method = std::make_shared<Fp8E4m3Method>(common_weight_loader_, tp_, quant_config);

  const size_t N = 4096;  // out features
  const size_t K = 320;   // in features
  const std::string o_prefix = "model.layers.0.self_attn.o_proj.";

  // Static FP8 weights
  Tensor weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP8_E4M3, {N, K});
  Tensor weight_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});
  Tensor input_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});

  fp8_method->load_attn_o_proj(device_model_weights_, o_prefix + "weight", weight, dev_rank_);
  fp8_method->load_attn_o_proj(device_model_weights_, o_prefix + "weight_scale", weight_scale, dev_rank_);
  fp8_method->load_attn_o_proj(device_model_weights_, o_prefix + "input_scale", input_scale, dev_rank_);

  // ColPara split for o_proj => dim=1 split
  EXPECT_EQ(device_model_weights_[o_prefix + "weight"].shape[1], K / tp_);

  fp8_method->process_attn_o_proj(device_model_weights_, o_prefix, dev_rank_);
  ExpectFp8ProcessedAndBound(device_model_weights_, o_prefix, /*expect_input_scale=*/true);
#endif
}

// Test for static FP8: load and process MLP gate/up projection
TEST_F(Fp8E4m3MethodTest, TestLoadAndProcessMlpGateUpProj_StaticFp8) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateStaticFp8QuantConfig();
  auto fp8_method = std::make_shared<Fp8E4m3Method>(common_weight_loader_, tp_, quant_config);

  const size_t N = 2560;  // intermediate_size
  const size_t K = 1024;  // hidden_size

  const std::string gate_prefix = "model.layers.0.mlp.gate_proj.";
  const std::string up_prefix = "model.layers.0.mlp.up_proj.";

  for (const auto& prefix : {gate_prefix, up_prefix}) {
    Tensor weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP8_E4M3, {N, K});
    Tensor weight_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});
    Tensor input_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});

    fp8_method->load_mlp_gate_up_proj(device_model_weights_, prefix + "weight", weight, dev_rank_);
    fp8_method->load_mlp_gate_up_proj(device_model_weights_, prefix + "weight_scale", weight_scale, dev_rank_);
    fp8_method->load_mlp_gate_up_proj(device_model_weights_, prefix + "input_scale", input_scale, dev_rank_);

    // RowPara split for gate/up_proj => dim=0 split
    EXPECT_EQ(device_model_weights_[prefix + "weight"].shape[0], N / tp_);
  }

  // Process gate_up_proj: merge gate and up, rescale, and bind scales
  fp8_method->process_mlp_gate_up_proj(device_model_weights_, gate_prefix, dev_rank_);

  const std::string gate_up_prefix = "model.layers.0.mlp.gate_up_proj.";
  ExpectFp8ProcessedAndBound(device_model_weights_, gate_up_prefix, /*expect_input_scale=*/true);

  // Original gate/up weight_scale and input_scale should be removed
  EXPECT_TRUE(device_model_weights_.find(gate_prefix + "weight_scale") == device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(up_prefix + "weight_scale") == device_model_weights_.end());
#endif
}

// Test for dynamic FP8: load and process MLP gate/up projection
TEST_F(Fp8E4m3MethodTest, TestLoadAndProcessMlpGateUpProj_DynamicFp8) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateDynamicFp8QuantConfig();
  auto fp8_method = std::make_shared<Fp8E4m3Method>(common_weight_loader_, tp_, quant_config);

  const size_t N = 2560;  // intermediate_size
  const size_t K = 1024;  // hidden_size

  const std::string gate_prefix = "model.layers.0.mlp.gate_proj.";
  const std::string up_prefix = "model.layers.0.mlp.up_proj.";

  for (const auto& prefix : {gate_prefix, up_prefix}) {
    // Dynamic FP8: load BF16 weights
    Tensor weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_BF16, {N, K});

    fp8_method->load_mlp_gate_up_proj(device_model_weights_, prefix + "weight", weight, dev_rank_);

    EXPECT_TRUE(device_model_weights_.find(prefix + "weight") != device_model_weights_.end());
    // RowPara split for gate/up_proj => dim=0 split
    EXPECT_EQ(device_model_weights_[prefix + "weight"].shape[0], N / tp_);
  }

  // Process gate_up_proj: merge and convert to FP8
  fp8_method->process_mlp_gate_up_proj(device_model_weights_, gate_prefix, dev_rank_);

  const std::string gate_up_prefix = "model.layers.0.mlp.gate_up_proj.";

  // After processing, merged weight should be FP8 type
  EXPECT_TRUE(device_model_weights_.find(gate_up_prefix + "weight") != device_model_weights_.end());
  EXPECT_EQ(device_model_weights_[gate_up_prefix + "weight"].dtype, DataType::TYPE_FP8_E4M3);
  EXPECT_TRUE(device_model_weights_[gate_up_prefix + "weight"].weight_scales != nullptr);
#endif
}

// Test for static FP8: load and process MLP down projection
TEST_F(Fp8E4m3MethodTest, TestLoadAndProcessMlpDownProj_StaticFp8) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateStaticFp8QuantConfig();
  auto fp8_method = std::make_shared<Fp8E4m3Method>(common_weight_loader_, tp_, quant_config);

  const size_t N = 9728;  // hidden_size
  const size_t K = 320;   // intermediate_size
  const std::string down_prefix = "model.layers.0.mlp.down_proj.";

  Tensor weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP8_E4M3, {N, K});
  Tensor weight_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});
  Tensor input_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});

  fp8_method->load_mlp_down_proj(device_model_weights_, down_prefix + "weight", weight, dev_rank_);
  fp8_method->load_mlp_down_proj(device_model_weights_, down_prefix + "weight_scale", weight_scale, dev_rank_);
  fp8_method->load_mlp_down_proj(device_model_weights_, down_prefix + "input_scale", input_scale, dev_rank_);

  // ColPara split for down_proj => dim=1 split
  EXPECT_EQ(device_model_weights_[down_prefix + "weight"].shape[1], K / tp_);

  fp8_method->process_mlp_down_proj(device_model_weights_, down_prefix, dev_rank_);
  ExpectFp8ProcessedAndBound(device_model_weights_, down_prefix, /*expect_input_scale=*/true);
#endif
}

// Test for dynamic FP8: load and process MLP down projection
TEST_F(Fp8E4m3MethodTest, TestLoadAndProcessMlpDownProj_DynamicFp8) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateDynamicFp8QuantConfig();
  auto fp8_method = std::make_shared<Fp8E4m3Method>(common_weight_loader_, tp_, quant_config);

  const size_t N = 9728;  // hidden_size
  const size_t K = 320;   // intermediate_size
  const std::string down_prefix = "model.layers.0.mlp.down_proj.";

  // Dynamic FP8: load BF16 weights
  Tensor weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_BF16, {N, K});

  fp8_method->load_mlp_down_proj(device_model_weights_, down_prefix + "weight", weight, dev_rank_);

  // ColPara split for down_proj => dim=1 split
  EXPECT_EQ(device_model_weights_[down_prefix + "weight"].shape[1], K / tp_);

  fp8_method->process_mlp_down_proj(device_model_weights_, down_prefix, dev_rank_);

  // After processing, weight should be FP8 type
  EXPECT_EQ(device_model_weights_[down_prefix + "weight"].dtype, DataType::TYPE_FP8_E4M3);
  EXPECT_TRUE(device_model_weights_[down_prefix + "weight"].weight_scales != nullptr);
#endif
}

// Test for static FP8: load and process experts projections (MOE)
TEST_F(Fp8E4m3MethodTest, TestLoadAndProcessExpertsProj_StaticFp8) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateStaticFp8QuantConfig();
  auto fp8_method = std::make_shared<Fp8E4m3Method>(common_weight_loader_, tp_, quant_config);

  const int num_experts = 4;
  const size_t inter_size = 256;
  const size_t hidden_units = 128;
  const std::string experts_prefix = "model.layers.0.mlp.experts.";

  // Load weights for each expert's gate_proj, up_proj, and down_proj using load_experts_proj
  for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    const std::string gate_proj_name = fmt::format("{}experts.{}.gate_proj.", "model.layers.0.mlp.", expert_idx);
    const std::string up_proj_name = fmt::format("{}experts.{}.up_proj.", "model.layers.0.mlp.", expert_idx);
    const std::string down_proj_name = fmt::format("{}experts.{}.down_proj.", "model.layers.0.mlp.", expert_idx);

    // Gate projection - using load_mlp_gate_up_proj
    Tensor gate_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP8_E4M3, {inter_size, hidden_units});
    Tensor gate_weight_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});
    Tensor gate_input_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});
    fp8_method->load_mlp_gate_up_proj(device_model_weights_, gate_proj_name + "weight", gate_weight, dev_rank_);
    fp8_method->load_mlp_gate_up_proj(device_model_weights_, gate_proj_name + "weight_scale", gate_weight_scale,
                                      dev_rank_);
    fp8_method->load_mlp_gate_up_proj(device_model_weights_, gate_proj_name + "input_scale", gate_input_scale,
                                      dev_rank_);

    // Up projection - using load_mlp_gate_up_proj
    Tensor up_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP8_E4M3, {inter_size, hidden_units});
    Tensor up_weight_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});
    Tensor up_input_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});
    fp8_method->load_mlp_gate_up_proj(device_model_weights_, up_proj_name + "weight", up_weight, dev_rank_);
    fp8_method->load_mlp_gate_up_proj(device_model_weights_, up_proj_name + "weight_scale", up_weight_scale, dev_rank_);
    fp8_method->load_mlp_gate_up_proj(device_model_weights_, up_proj_name + "input_scale", up_input_scale, dev_rank_);

    // Down projection - using load_mlp_gate_up_proj
    Tensor down_weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP8_E4M3, {hidden_units, inter_size});
    Tensor down_weight_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});
    Tensor down_input_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1});
    fp8_method->load_mlp_gate_up_proj(device_model_weights_, down_proj_name + "weight", down_weight, dev_rank_);
    fp8_method->load_mlp_gate_up_proj(device_model_weights_, down_proj_name + "weight_scale", down_weight_scale,
                                      dev_rank_);
    fp8_method->load_mlp_gate_up_proj(device_model_weights_, down_proj_name + "input_scale", down_input_scale,
                                      dev_rank_);
  }

  // Process experts projections
  const std::string weight_prefix = "model.layers.0.mlp.experts.";
  fp8_method->process_experts_proj(device_model_weights_, weight_prefix, dev_rank_, num_experts);

  // Verify merged up_gate_proj
  const std::string up_gate_weight_name = weight_prefix + "up_gate_proj.weight";
  const std::string up_gate_weight_scale_name = weight_prefix + "up_gate_proj.weight_scale";
  const std::string up_gate_input_scale_name = weight_prefix + "up_gate_proj.input_scale";

  EXPECT_TRUE(device_model_weights_.find(up_gate_weight_name) != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(up_gate_weight_scale_name) != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(up_gate_input_scale_name) != device_model_weights_.end());

  // Verify shape: [num_experts, 2 * inter_size / tp_, hidden_units]
  // gate/up proj uses RowPara, split on dim=0, after merge becomes dim=1
  EXPECT_EQ(device_model_weights_[up_gate_weight_name].shape.size(), 3);
  EXPECT_EQ(device_model_weights_[up_gate_weight_name].shape[0], static_cast<size_t>(num_experts));
  EXPECT_EQ(device_model_weights_[up_gate_weight_name].shape[1], 2 * inter_size / tp_);
  EXPECT_EQ(device_model_weights_[up_gate_weight_name].shape[2], hidden_units);

  // Verify weight_scale shape: [num_experts]
  EXPECT_EQ(device_model_weights_[up_gate_weight_scale_name].shape[0], static_cast<size_t>(num_experts));

  // Verify scales are bound
  EXPECT_TRUE(device_model_weights_[up_gate_weight_name].weight_scales != nullptr);
  EXPECT_TRUE(device_model_weights_[up_gate_weight_name].input_scales != nullptr);

  // Verify merged down_proj
  const std::string down_weight_name = weight_prefix + "down_proj.weight";
  const std::string down_weight_scale_name = weight_prefix + "down_proj.weight_scale";
  const std::string down_input_scale_name = weight_prefix + "down_proj.input_scale";

  EXPECT_TRUE(device_model_weights_.find(down_weight_name) != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(down_weight_scale_name) != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(down_input_scale_name) != device_model_weights_.end());

  // Verify shape: [num_experts, hidden_units, inter_size / tp_]
  // down proj uses RowPara (via load_mlp_gate_up_proj), split on dim=0, becomes dim=2 after reshape
  EXPECT_EQ(device_model_weights_[down_weight_name].shape.size(), 3);
  EXPECT_EQ(device_model_weights_[down_weight_name].shape[0], static_cast<size_t>(num_experts));
  EXPECT_EQ(device_model_weights_[down_weight_name].shape[1], hidden_units / tp_);
  EXPECT_EQ(device_model_weights_[down_weight_name].shape[2], inter_size);

  // Verify weight_scale shape: [num_experts]
  EXPECT_EQ(device_model_weights_[down_weight_scale_name].shape[0], static_cast<size_t>(num_experts));

  // Verify scales are bound
  EXPECT_TRUE(device_model_weights_[down_weight_name].weight_scales != nullptr);
  EXPECT_TRUE(device_model_weights_[down_weight_name].input_scales != nullptr);

  // Verify original expert weights are removed
  for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    const std::string gate_proj_name = fmt::format("{}experts.{}.gate_proj.", "model.layers.0.mlp.", expert_idx);
    const std::string up_proj_name = fmt::format("{}experts.{}.up_proj.", "model.layers.0.mlp.", expert_idx);
    const std::string down_proj_name = fmt::format("{}experts.{}.down_proj.", "model.layers.0.mlp.", expert_idx);

    EXPECT_TRUE(device_model_weights_.find(gate_proj_name + "weight_scale") == device_model_weights_.end());
    EXPECT_TRUE(device_model_weights_.find(up_proj_name + "weight_scale") == device_model_weights_.end());
    EXPECT_TRUE(device_model_weights_.find(down_proj_name + "weight_scale") == device_model_weights_.end());
    EXPECT_TRUE(device_model_weights_.find(gate_proj_name + "input_scale") == device_model_weights_.end());
    EXPECT_TRUE(device_model_weights_.find(up_proj_name + "input_scale") == device_model_weights_.end());
    EXPECT_TRUE(device_model_weights_.find(down_proj_name + "input_scale") == device_model_weights_.end());
  }
#endif
}
