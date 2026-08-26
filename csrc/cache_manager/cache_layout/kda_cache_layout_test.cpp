/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache_manager/cache_layout/cache_layout_factory.h"
#include "cache_manager/cache_layout/kda_cache_layout.h"
#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "device/device_types.h"

namespace ksana_llm {

namespace {
PipelineConfig MakePipelineConfig(uint32_t num_layer) {
  PipelineConfig cfg;
  cfg.lower_layer_idx = 0;
  cfg.upper_layer_idx = static_cast<int16_t>(num_layer - 1);
  cfg.lower_nextn_layer_idx = -1;
  cfg.upper_nextn_layer_idx = -1;
  return cfg;
}

RuntimeConfig MakeRuntimeConfig(size_t tp_size = 1) {
  RuntimeConfig cfg;
  cfg.attn_backend_config.block_token_num = 16;
  cfg.attn_backend_config.kv_cache_dtype = TYPE_FP16;
  cfg.parallel_basic_config.tensor_parallel_size = tp_size;
  cfg.parallel_basic_config.attn_tensor_parallel_size = tp_size;
  return cfg;
}

// 4 heads × 128, conv kernel 4, kda_layers={1,3} (1-based) → 0-based {0,2}.
// KDA has no GQA, so num_kv_heads is not set (== num_heads).
ModelConfig MakeKdaModelConfig() {
  ModelConfig cfg;
  cfg.type = "kimi_k3";
  cfg.num_layer = 4;
  cfg.weight_data_type = TYPE_FP16;
  cfg.use_linear_attn = true;
  cfg.head_num = 4;
  cfg.size_per_head = 128;
  cfg.kimi_linear_config.head_dim = 128;
  cfg.kimi_linear_config.short_conv_kernel_size = 4;
  cfg.kimi_linear_config.kda_layers = {1, 3};
  return cfg;
}

// num_layer-layer hybrid (8 heads × 128, conv kernel 4): odd 1-based layers are
// KDA. Used by the factory integration test.
ModelConfig MakeKdaHybridModelConfig(uint32_t num_layer) {
  ModelConfig cfg;
  cfg.type = "kimi_k3";
  cfg.num_layer = num_layer;
  cfg.weight_data_type = TYPE_FP16;
  cfg.use_linear_attn = true;
  cfg.head_num = 8;
  cfg.size_per_head = 128;
  cfg.kimi_linear_config.head_dim = 128;
  cfg.kimi_linear_config.short_conv_kernel_size = 4;
  for (uint32_t l = 1; l <= num_layer; l += 2) {
    cfg.kimi_linear_config.kda_layers.insert(l);
  }
  return cfg;
}
}  // namespace

// Fixture: default config is 4 heads × 128, conv kernel 4, kda_layers={1,3}, tp=1,
// kv_cache_dtype=fp16. Per-KDA-layer: conv=9216 (bf16) + recurrent=262144 (fp32)
// = 271360; 2 KDA layers → block=542720. Tests needing a variant mutate runtime_cfg_.
class KDAStateLayoutTest : public ::testing::Test {
 protected:
  ModelConfig model_cfg_ = MakeKdaModelConfig();
  PipelineConfig pipeline_cfg_ = MakePipelineConfig(model_cfg_.num_layer);
  RuntimeConfig runtime_cfg_ = MakeRuntimeConfig();
};

TEST_F(KDAStateLayoutTest, ComputesConvAndRecurrentSizes) {
  KDAStateLayout layout(model_cfg_, pipeline_cfg_, runtime_cfg_);
  EXPECT_EQ(layout.GetBlockCacheSize(), 542720u);
  EXPECT_EQ(layout.GetActiveKvCacheTypes(), (std::vector<std::string>{"conv", "recurrent"}));
  // Layer 0 (KDA): conv at 0, recurrent at conv_size=9216; k/v share each offset.
  EXPECT_EQ(layout.GetKvCacheLayerOffsets(0, "conv").k_offset, 0u);
  EXPECT_EQ(layout.GetKvCacheLayerOffsets(0, "recurrent").k_offset, 9216u);
  // Layer 2 (KDA): base=271360, recurrent at base+9216=280576.
  EXPECT_EQ(layout.GetKvCacheLayerOffsets(2, "conv").k_offset, 271360u);
  EXPECT_EQ(layout.GetKvCacheLayerOffsets(2, "recurrent").k_offset, 280576u);
}

TEST_F(KDAStateLayoutTest, InactiveLayerGetsZeroOffset) {
  // Layers 1 and 3 are full-attn (not in kda_layers): zero-offset placeholder.
  KDAStateLayout layout(model_cfg_, pipeline_cfg_, runtime_cfg_);
  EXPECT_EQ(layout.GetKvCacheLayerOffsets(1, "conv").k_offset, 0u);
  EXPECT_EQ(layout.GetKvCacheLayerOffsets(3, "recurrent").k_offset, 0u);
}

TEST_F(KDAStateLayoutTest, TpShardingHalvesHeads) {
  // tp=2: conv_size=4608, recurrent=131072; per-KDA-layer=135680; block=271360.
  runtime_cfg_.parallel_basic_config.tensor_parallel_size = 2;
  runtime_cfg_.parallel_basic_config.attn_tensor_parallel_size = 2;
  KDAStateLayout layout(model_cfg_, pipeline_cfg_, runtime_cfg_);
  EXPECT_EQ(layout.GetBlockCacheSize(), 2u * (4608u + 131072u));
  EXPECT_EQ(layout.GetKvCacheLayerOffsets(0, "recurrent").k_offset, 4608u);
}

TEST_F(KDAStateLayoutTest, DtypesAreIndependentOfKvCacheDtype) {
  // conv is fixed bf16; recurrent defaults fp32 — both unaffected by kv_cache_dtype.
  runtime_cfg_.attn_backend_config.kv_cache_dtype = TYPE_FP32;
  KDAStateLayout layout(model_cfg_, pipeline_cfg_, runtime_cfg_);
  EXPECT_EQ(layout.GetBlockCacheSize(), 542720u);
  EXPECT_EQ(layout.GetKvCacheLayerOffsets(0, "recurrent").k_offset, 9216u);

  // Overriding recurrent to bf16 halves it (262144 → 131072); conv unchanged.
  KDAStateLayout bf16_rec(model_cfg_, pipeline_cfg_, runtime_cfg_, TYPE_BF16);
  EXPECT_EQ(bf16_rec.GetBlockCacheSize(), 2u * (9216u + 131072u));
  EXPECT_EQ(bf16_rec.GetKvCacheLayerOffsets(0, "recurrent").k_offset, 9216u);
}

TEST_F(KDAStateLayoutTest, ThrowsOnInvalidLayerOrType) {
  KDAStateLayout layout(model_cfg_, pipeline_cfg_, runtime_cfg_);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(4, "conv"), std::runtime_error);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(-1, "recurrent"), std::runtime_error);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(0, "attention"), std::runtime_error);
}

// CacheLayoutFactory wires KDAStateLayout into "mamba-state" when kda_layers is
// non-empty (10-layer hybrid, 5 KDA layers). conv=18432, recurrent=524288;
// per-KDA-layer=542720; block=5*542720=2713600.
TEST(KDAStateLayoutFactoryTest, SizedViaFactory) {
  auto model_cfg = MakeKdaHybridModelConfig(/*num_layer=*/10);
  const auto pipeline_cfg = MakePipelineConfig(model_cfg.num_layer);
  const auto runtime_cfg = MakeRuntimeConfig();

  auto layout = CacheLayoutFactory::CreateModelCacheLayout(model_cfg, pipeline_cfg, runtime_cfg);
  ASSERT_TRUE(layout->IsBlockCacheTypeExist("mamba-state"));
  auto kda_layout = std::dynamic_pointer_cast<KDAStateLayout>(layout->GetBlockCacheLayout("mamba-state"));
  ASSERT_NE(kda_layout, nullptr);
  EXPECT_EQ(kda_layout->GetBlockCacheSize(), 5u * (18432u + 524288u));
  EXPECT_EQ(kda_layout->GetActiveKvCacheTypes(), (std::vector<std::string>{"conv", "recurrent"}));
}

}  // namespace ksana_llm
