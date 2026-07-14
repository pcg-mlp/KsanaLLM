/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/weight_method/awq_method.h"

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

static inline QuantConfig CreateAwqQuantConfig(const LinearComputeBackend& backend) {
  QuantConfig qc;
  qc.method = QUANT_AWQ;
  qc.bits = 4;
  qc.group_size = 128;
  qc.backend = backend;
  return qc;
}

}  // namespace

class AwqMethodTest : public testing::Test {
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

static void ExpectAwqProcessedAndBound(std::unordered_map<std::string, Tensor>& weights, const std::string& prefix) {
  const std::string scales_name = prefix + "scales";
  const std::string zeros_name = prefix + "zeros";
  const std::string weight_name = prefix + "weight";

  EXPECT_TRUE(weights.find(weight_name) != weights.end());
  EXPECT_TRUE(weights.find(scales_name) != weights.end());
  EXPECT_TRUE(weights.find(zeros_name) != weights.end());

  EXPECT_TRUE(weights.find(prefix + "qweight") == weights.end());
  EXPECT_TRUE(weights.find(prefix + "qzeros") == weights.end());

  EXPECT_TRUE(weights[weight_name].scales != nullptr);
  EXPECT_TRUE(weights[weight_name].zeros != nullptr);
}

TEST_F(AwqMethodTest, TestLoadAndProcessAttnQKVProj_Cutlass) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateAwqQuantConfig(CUTLASS_LINEAR_BACKEND);
  auto awq_method = std::make_shared<AwqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 256;           // out features
  const size_t K = 128;           // in features
  const size_t group_size = 128;  // AWQ group size

  const std::string q_prefix = "model.layers.0.self_attn.q_proj.";
  const std::string k_prefix = "model.layers.0.self_attn.k_proj.";
  const std::string v_prefix = "model.layers.0.self_attn.v_proj.";

  for (const auto& prefix : {q_prefix, k_prefix, v_prefix}) {
    // AWQ weights are packed: 8 x 4-bit values per int32.
    Tensor qweight(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {N, K / 8});
    Tensor qzeros(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {N / group_size, K / 8});
    Tensor scales(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N / group_size, K});

    awq_method->load_attn_q_k_v_proj(device_model_weights_, prefix + "qweight", qweight, dev_rank_);
    awq_method->load_attn_q_k_v_proj(device_model_weights_, prefix + "qzeros", qzeros, dev_rank_);
    awq_method->load_attn_q_k_v_proj(device_model_weights_, prefix + "scales", scales, dev_rank_);

    EXPECT_TRUE(device_model_weights_.find(prefix + "qweight") != device_model_weights_.end());
    EXPECT_TRUE(device_model_weights_.find(prefix + "qzeros") != device_model_weights_.end());
    EXPECT_TRUE(device_model_weights_.find(prefix + "scales") != device_model_weights_.end());

    // After preprocessing: qweight shape becomes [N, K/2] (packed int4), qzeros becomes [N/group_size, K] (fp16).
    EXPECT_EQ(device_model_weights_[prefix + "qweight"].shape[1], (K / 2) / tp_);
    EXPECT_EQ(device_model_weights_[prefix + "qzeros"].shape[1], K / tp_);
    EXPECT_EQ(device_model_weights_[prefix + "scales"].shape[1], K / tp_);
  }

  // process qkv: merge and convert/bind.
  awq_method->process_attn_qkv_proj(device_model_weights_, q_prefix, dev_rank_);

  const std::string qkv_prefix = "model.layers.0.self_attn.query_key_value.";
  ExpectAwqProcessedAndBound(device_model_weights_, qkv_prefix);
#endif
}

TEST_F(AwqMethodTest, TestLoadAndProcessMlpGateUpProj_Marlin) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateAwqQuantConfig(MARLIN_LINEAR_BACKEND);
  auto awq_method = std::make_shared<AwqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 2560;
  const size_t K = 1216;
  const size_t group_size = 128;

  const std::string gate_prefix = "model.layers.0.mlp.gate_proj.";
  const std::string up_prefix = "model.layers.0.mlp.up_proj.";

  for (const auto& prefix : {gate_prefix, up_prefix}) {
    // AWQ weights are packed: 8 x 4-bit values per int32.
    Tensor qweight(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {N, K / 8});
    Tensor qzeros(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {N / group_size, K / 8});
    Tensor scales(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N / group_size, K});

    awq_method->load_mlp_gate_up_proj(device_model_weights_, prefix + "qweight", qweight, dev_rank_);
    awq_method->load_mlp_gate_up_proj(device_model_weights_, prefix + "qzeros", qzeros, dev_rank_);
    awq_method->load_mlp_gate_up_proj(device_model_weights_, prefix + "scales", scales, dev_rank_);

    // Marlin backend: no preprocessing, keeps AWQ packed format [N, K/8].
    // RowPara split for gate/up_proj => dim=1 split.
    EXPECT_EQ(device_model_weights_[prefix + "qweight"].shape[1], (K / 8) / tp_);
    EXPECT_EQ(device_model_weights_[prefix + "qzeros"].shape[1], (K / 8) / tp_);
    EXPECT_EQ(device_model_weights_[prefix + "scales"].shape[1], K / tp_);
  }

  awq_method->process_mlp_gate_up_proj(device_model_weights_, gate_prefix, dev_rank_);
  ExpectAwqProcessedAndBound(device_model_weights_, gate_prefix);

  awq_method->process_mlp_gate_up_proj(device_model_weights_, up_prefix, dev_rank_);
  ExpectAwqProcessedAndBound(device_model_weights_, up_prefix);
#endif
}

TEST_F(AwqMethodTest, TestLoadAndProcessAttnOProj_Cutlass) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateAwqQuantConfig(CUTLASS_LINEAR_BACKEND);
  auto awq_method = std::make_shared<AwqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 4096;
  const size_t K = 320;
  const size_t group_size = 128;
  const std::string o_prefix = "model.layers.0.self_attn.o_proj.";

  // AWQ weights are packed: 8 x 4-bit values per int32.
  Tensor qweight(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {N, K / 8});
  Tensor qzeros(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {N / group_size, K / 8});
  Tensor scales(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N / group_size, K});

  awq_method->load_attn_o_proj(device_model_weights_, o_prefix + "qweight", qweight, dev_rank_);
  awq_method->load_attn_o_proj(device_model_weights_, o_prefix + "qzeros", qzeros, dev_rank_);
  awq_method->load_attn_o_proj(device_model_weights_, o_prefix + "scales", scales, dev_rank_);

  // After preprocessing: qweight shape becomes [N, K/2] (packed int4), qzeros becomes [N/group_size, K] (fp16).
  EXPECT_EQ(device_model_weights_[o_prefix + "qweight"].shape[0], N / tp_);
  EXPECT_EQ(device_model_weights_[o_prefix + "qzeros"].shape[0], (N / group_size) / tp_);
  EXPECT_EQ(device_model_weights_[o_prefix + "scales"].shape[0], (N / group_size) / tp_);

  awq_method->process_attn_o_proj(device_model_weights_, o_prefix, dev_rank_);
  ExpectAwqProcessedAndBound(device_model_weights_, o_prefix);
#endif
}

TEST_F(AwqMethodTest, TestLoadAndProcessMlpDownProj_Marlin) {
#ifdef ENABLE_CUDA
  const auto quant_config = CreateAwqQuantConfig(MARLIN_LINEAR_BACKEND);
  auto awq_method = std::make_shared<AwqMethod>(common_weight_loader_, tp_, quant_config);

  const size_t N = 9728;
  const size_t K = 320;
  const size_t group_size = 128;
  const std::string down_prefix = "model.layers.0.mlp.down_proj.";

  // AWQ weights are packed: 8 x 4-bit values per int32.
  Tensor qweight(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {N, K / 8});
  Tensor qzeros(MemoryLocation::LOCATION_HOST, DataType::TYPE_INT32, {N / group_size, K / 8});
  Tensor scales(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP16, {N / group_size, K});

  awq_method->load_mlp_down_proj(device_model_weights_, down_prefix + "qweight", qweight, dev_rank_);
  awq_method->load_mlp_down_proj(device_model_weights_, down_prefix + "qzeros", qzeros, dev_rank_);
  awq_method->load_mlp_down_proj(device_model_weights_, down_prefix + "scales", scales, dev_rank_);

  // RowPara split for down_proj => dim=0 split.
  EXPECT_EQ(device_model_weights_[down_prefix + "qweight"].shape[0], N / tp_);
  EXPECT_EQ(device_model_weights_[down_prefix + "qzeros"].shape[0], (N / group_size) / tp_);
  EXPECT_EQ(device_model_weights_[down_prefix + "scales"].shape[0], (N / group_size) / tp_);

  awq_method->process_mlp_down_proj(device_model_weights_, down_prefix, dev_rank_);
  ExpectAwqProcessedAndBound(device_model_weights_, down_prefix);
#endif
}
