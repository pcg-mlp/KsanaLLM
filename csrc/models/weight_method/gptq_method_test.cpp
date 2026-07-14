/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/weight_method/gptq_method.h"

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

static inline QuantConfig CreateGptqQuantConfig(const LinearComputeBackend& backend, bool desc_act = false) {
  QuantConfig qc;
  qc.method = QUANT_GPTQ;
  qc.bits = 4;
  qc.group_size = 128;
  qc.backend = backend;
  qc.desc_act = desc_act;
  return qc;
}

}  // namespace

class GptqMethodTest : public testing::Test {
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

static void ExpectGptqProcessedAndBound(std::unordered_map<std::string, Tensor>& weights, const std::string& prefix) {
  const std::string scales_name = prefix + "scales";
  const std::string weight_name = prefix + "weight";

  EXPECT_TRUE(weights.find(weight_name) != weights.end()) << "Weight not found: " << weight_name;
  EXPECT_TRUE(weights.find(scales_name) != weights.end()) << "Scales not found: " << scales_name;

  EXPECT_TRUE(weights.find(prefix + "qweight") == weights.end()) << "qweight should be removed after processing";

  EXPECT_TRUE(weights[weight_name].scales != nullptr) << "Scales not bound for: " << weight_name;
}

TEST_F(GptqMethodTest, TestLoadAndProcessAttnQKVProj_Cutlass) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateGptqQuantConfig(CUTLASS_LINEAR_BACKEND);
  auto gptq_method = std::make_shared<GptqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 2048;          // out features
  const size_t K = 128;           // in features
  const size_t group_size = 128;  // GPTQ group size

  const std::string q_prefix = "model.layers.0.self_attn.q_proj.";
  const std::string k_prefix = "model.layers.0.self_attn.k_proj.";
  const std::string v_prefix = "model.layers.0.self_attn.v_proj.";

  for (const auto& prefix : {q_prefix, k_prefix, v_prefix}) {
    // GPTQ weights are packed: 8 x 4-bit values per int32.
    Tensor qweight(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {K / 8, N});
    Tensor scales(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {K / group_size, N});

    gptq_method->load_attn_q_k_v_proj(device_model_weights_, prefix + "qweight", qweight, dev_rank_);
    gptq_method->load_attn_q_k_v_proj(device_model_weights_, prefix + "scales", scales, dev_rank_);

    EXPECT_TRUE(device_model_weights_.find(prefix + "qweight") != device_model_weights_.end());
    EXPECT_TRUE(device_model_weights_.find(prefix + "scales") != device_model_weights_.end());

    // After preprocessing for CUTLASS: qweight shape becomes [K, N / 2] (packed int4).
    // ColPara split for qkv_proj => dim=1 split.
    EXPECT_EQ(device_model_weights_[prefix + "qweight"].shape[1], (N / 2) / tp_);
    EXPECT_EQ(device_model_weights_[prefix + "scales"].shape[1], N / tp_);
  }

  // process qkv: merge and convert/bind.
  gptq_method->process_attn_qkv_proj(device_model_weights_, q_prefix, dev_rank_);

  const std::string qkv_prefix = "model.layers.0.self_attn.query_key_value.";
  ExpectGptqProcessedAndBound(device_model_weights_, qkv_prefix);
#endif
}

TEST_F(GptqMethodTest, TestLoadAndProcessAttnQKVProj_Marlin) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateGptqQuantConfig(MARLIN_LINEAR_BACKEND);
  auto gptq_method = std::make_shared<GptqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 256;
  const size_t K = 128;
  const size_t group_size = 128;

  const std::string q_prefix = "model.layers.0.self_attn.q_proj.";
  const std::string k_prefix = "model.layers.0.self_attn.k_proj.";
  const std::string v_prefix = "model.layers.0.self_attn.v_proj.";

  for (const auto& prefix : {q_prefix, k_prefix, v_prefix}) {
    // GPTQ weights are packed: 8 x 4-bit values per int32.
    Tensor qweight(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {N, K / 8});
    Tensor scales(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N / group_size, K});

    gptq_method->load_attn_q_k_v_proj(device_model_weights_, prefix + "qweight", qweight, dev_rank_);
    gptq_method->load_attn_q_k_v_proj(device_model_weights_, prefix + "scales", scales, dev_rank_);

    // Marlin backend: no preprocessing before split, keeps GPTQ packed format [N, K/8].
    // ColPara split for qkv_proj => dim=1 split.
    EXPECT_EQ(device_model_weights_[prefix + "qweight"].shape[1], (K / 8) / tp_);
    EXPECT_EQ(device_model_weights_[prefix + "scales"].shape[1], K / tp_);
  }

  gptq_method->process_attn_qkv_proj(device_model_weights_, q_prefix, dev_rank_);

  const std::string qkv_prefix = "model.layers.0.self_attn.query_key_value.";
  ExpectGptqProcessedAndBound(device_model_weights_, qkv_prefix);
#endif
}

TEST_F(GptqMethodTest, TestLoadAndProcessAttnOProj_Cutlass) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateGptqQuantConfig(CUTLASS_LINEAR_BACKEND);
  auto gptq_method = std::make_shared<GptqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 2048;
  const size_t K = 4096;
  const size_t group_size = 128;
  const std::string o_prefix = "model.layers.0.self_attn.o_proj.";

  // GPTQ weights are packed: 8 x 4-bit values per int32.
  Tensor qweight(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {K / 8, N});
  Tensor scales(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {K / group_size, N});

  gptq_method->load_attn_o_proj(device_model_weights_, o_prefix + "qweight", qweight, dev_rank_);
  gptq_method->load_attn_o_proj(device_model_weights_, o_prefix + "scales", scales, dev_rank_);

  // After preprocessing for CUTLASS: qweight shape becomes [K, N / 2] (packed int4).
  // RowPara split for o_proj => dim=0 split.
  EXPECT_EQ(device_model_weights_[o_prefix + "qweight"].shape[0], K / tp_);
  EXPECT_EQ(device_model_weights_[o_prefix + "scales"].shape[0], (K / group_size) / tp_);

  gptq_method->process_attn_o_proj(device_model_weights_, o_prefix, dev_rank_);
  ExpectGptqProcessedAndBound(device_model_weights_, o_prefix);
#endif
}

TEST_F(GptqMethodTest, TestLoadAndProcessAttnOProj_Marlin) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateGptqQuantConfig(MARLIN_LINEAR_BACKEND);
  auto gptq_method = std::make_shared<GptqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 4096;
  const size_t K = 320;
  const size_t group_size = 128;
  const std::string o_prefix = "model.layers.0.self_attn.o_proj.";

  Tensor qweight(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {N, K / 8});
  Tensor scales(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N / group_size, K});

  gptq_method->load_attn_o_proj(device_model_weights_, o_prefix + "qweight", qweight, dev_rank_);
  gptq_method->load_attn_o_proj(device_model_weights_, o_prefix + "scales", scales, dev_rank_);

  // Marlin backend: no preprocessing, keeps GPTQ packed format [N, K/8].
  // RowPara split for o_proj => dim=0 split.
  EXPECT_EQ(device_model_weights_[o_prefix + "qweight"].shape[0], N / tp_);
  EXPECT_EQ(device_model_weights_[o_prefix + "scales"].shape[0], (N / group_size) / tp_);

  gptq_method->process_attn_o_proj(device_model_weights_, o_prefix, dev_rank_);
  ExpectGptqProcessedAndBound(device_model_weights_, o_prefix);
#endif
}

TEST_F(GptqMethodTest, TestLoadAndProcessMlpGateUpProj_Cutlass) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateGptqQuantConfig(CUTLASS_LINEAR_BACKEND);
  auto gptq_method = std::make_shared<GptqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 12288;
  const size_t K = 4096;
  const size_t group_size = 128;

  const std::string gate_prefix = "model.layers.0.mlp.gate_proj.";
  const std::string up_prefix = "model.layers.0.mlp.up_proj.";

  for (const auto& prefix : {gate_prefix, up_prefix}) {
    // GPTQ weights are packed: 8 x 4-bit values per int32.
    Tensor qweight(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {K / 8, N});
    Tensor scales(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {K / group_size, N});

    gptq_method->load_mlp_gate_up_proj(device_model_weights_, prefix + "qweight", qweight, dev_rank_);
    gptq_method->load_mlp_gate_up_proj(device_model_weights_, prefix + "scales", scales, dev_rank_);

    // After preprocessing for CUTLASS: qweight shape becomes [K, N / 2] (packed int4).
    // ColPara split for gate/up_proj => dim=1 split.
    EXPECT_EQ(device_model_weights_[prefix + "qweight"].shape[1], (N / 2) / tp_);
    EXPECT_EQ(device_model_weights_[prefix + "scales"].shape[1], N / tp_);
  }

  gptq_method->process_mlp_gate_up_proj(device_model_weights_, gate_prefix, dev_rank_);
  ExpectGptqProcessedAndBound(device_model_weights_, gate_prefix);

  gptq_method->process_mlp_gate_up_proj(device_model_weights_, up_prefix, dev_rank_);
  ExpectGptqProcessedAndBound(device_model_weights_, up_prefix);
#endif
}

TEST_F(GptqMethodTest, TestLoadAndProcessMlpGateUpProj_Marlin) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateGptqQuantConfig(MARLIN_LINEAR_BACKEND);
  auto gptq_method = std::make_shared<GptqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 2560;
  const size_t K = 1216;
  const size_t group_size = 128;

  const std::string gate_prefix = "model.layers.0.mlp.gate_proj.";
  const std::string up_prefix = "model.layers.0.mlp.up_proj.";

  for (const auto& prefix : {gate_prefix, up_prefix}) {
    Tensor qweight(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {N, K / 8});
    Tensor scales(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N / group_size, K});

    gptq_method->load_mlp_gate_up_proj(device_model_weights_, prefix + "qweight", qweight, dev_rank_);
    gptq_method->load_mlp_gate_up_proj(device_model_weights_, prefix + "scales", scales, dev_rank_);

    // Marlin backend: no preprocessing, keeps GPTQ packed format [N, K/8].
    // ColPara split for gate/up_proj => dim=1 split.
    EXPECT_EQ(device_model_weights_[prefix + "qweight"].shape[1], (K / 8) / tp_);
    EXPECT_EQ(device_model_weights_[prefix + "scales"].shape[1], K / tp_);
  }

  gptq_method->process_mlp_gate_up_proj(device_model_weights_, gate_prefix, dev_rank_);
  ExpectGptqProcessedAndBound(device_model_weights_, gate_prefix);

  gptq_method->process_mlp_gate_up_proj(device_model_weights_, up_prefix, dev_rank_);
  ExpectGptqProcessedAndBound(device_model_weights_, up_prefix);
#endif
}

TEST_F(GptqMethodTest, TestLoadAndProcessMlpDownProj_Cutlass) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateGptqQuantConfig(CUTLASS_LINEAR_BACKEND);
  auto gptq_method = std::make_shared<GptqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 4096;
  const size_t K = 12288;
  const size_t group_size = 128;
  const std::string down_prefix = "model.layers.0.mlp.down_proj.";

  // GPTQ weights are packed: 8 x 4-bit values per int32.
  Tensor qweight(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {K / 8, N});
  Tensor scales(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {K / group_size, N});

  gptq_method->load_mlp_down_proj(device_model_weights_, down_prefix + "qweight", qweight, dev_rank_);
  gptq_method->load_mlp_down_proj(device_model_weights_, down_prefix + "scales", scales, dev_rank_);

  // After preprocessing for CUTLASS: qweight shape becomes [K, K/2] (packed int4).
  // RowPara split for down_proj => dim=0 split.
  EXPECT_EQ(device_model_weights_[down_prefix + "qweight"].shape[0], K / tp_);
  EXPECT_EQ(device_model_weights_[down_prefix + "scales"].shape[0], (K / group_size) / tp_);

  gptq_method->process_mlp_down_proj(device_model_weights_, down_prefix, dev_rank_);
  ExpectGptqProcessedAndBound(device_model_weights_, down_prefix);
#endif
}

TEST_F(GptqMethodTest, TestLoadAndProcessMlpDownProj_Marlin) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateGptqQuantConfig(MARLIN_LINEAR_BACKEND);
  auto gptq_method = std::make_shared<GptqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 9728;
  const size_t K = 320;
  const size_t group_size = 128;
  const std::string down_prefix = "model.layers.0.mlp.down_proj.";

  Tensor qweight(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {N, K / 8});
  Tensor scales(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N / group_size, K});

  gptq_method->load_mlp_down_proj(device_model_weights_, down_prefix + "qweight", qweight, dev_rank_);
  gptq_method->load_mlp_down_proj(device_model_weights_, down_prefix + "scales", scales, dev_rank_);

  // Marlin backend: no preprocessing, keeps GPTQ packed format [N, K/8].
  // RowPara split for down_proj => dim=0 split.
  EXPECT_EQ(device_model_weights_[down_prefix + "qweight"].shape[0], N / tp_);
  EXPECT_EQ(device_model_weights_[down_prefix + "scales"].shape[0], (N / group_size) / tp_);

  gptq_method->process_mlp_down_proj(device_model_weights_, down_prefix, dev_rank_);
  ExpectGptqProcessedAndBound(device_model_weights_, down_prefix);
#endif
}

TEST_F(GptqMethodTest, TestBiasHandling) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateGptqQuantConfig(CUTLASS_LINEAR_BACKEND);
  auto gptq_method = std::make_shared<GptqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 256;
  const std::string q_prefix = "model.layers.0.self_attn.q_proj.";

  // Test bias loading for q_proj
  Tensor bias(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N});

  gptq_method->load_attn_q_k_v_proj(device_model_weights_, q_prefix + "bias", bias, dev_rank_);

  EXPECT_TRUE(device_model_weights_.find(q_prefix + "bias") != device_model_weights_.end());
  // RowPara split for bias => dim=0 split.
  EXPECT_EQ(device_model_weights_[q_prefix + "bias"].shape[0], N / tp_);
#endif
}

TEST_F(GptqMethodTest, TestScalesDtypeSupport) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateGptqQuantConfig(MARLIN_LINEAR_BACKEND);
  auto gptq_method = std::make_shared<GptqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 256;
  const size_t K = 128;
  const size_t group_size = 128;
  const std::string q_prefix = "model.layers.0.self_attn.q_proj.";

  // Test with different scale dtypes (FP32, FP16, BF16)
  for (const auto& dtype : {DataType::TYPE_FP32, DataType::TYPE_FP16, DataType::TYPE_BF16}) {
    device_model_weights_.clear();

    Tensor qweight(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {N, K / 8});
    Tensor scales(MemoryLocation::LOCATION_HOST, dtype, {N / group_size, K});

    gptq_method->load_attn_q_k_v_proj(device_model_weights_, q_prefix + "qweight", qweight, dev_rank_);
    gptq_method->load_attn_q_k_v_proj(device_model_weights_, q_prefix + "scales", scales, dev_rank_);

    EXPECT_TRUE(device_model_weights_.find(q_prefix + "qweight") != device_model_weights_.end());
    EXPECT_TRUE(device_model_weights_.find(q_prefix + "scales") != device_model_weights_.end());
    EXPECT_EQ(device_model_weights_[q_prefix + "scales"].dtype, dtype);
  }
#endif
}
