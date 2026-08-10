/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "cache_manager/cache_layout/cache_layout_factory.h"
#include "cache_manager/cache_layout/mamba_cache_layout.h"
#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "device/device_types.h"

namespace ksana_llm {

namespace {

// Generic hybrid ModelConfig for linear-attn models. linear-attn state 的真实
// 尺寸待 linear-attn 布局单独实现，mamba-state block 当前占位 0，测试只需
// 一个 use_linear_attn 配置即可，不依赖具体几何。
ModelConfig MakeMambaModelConfig(uint32_t num_layer) {
  ModelConfig cfg;
  cfg.type = "qwen3_5";  // 线性 attn 模型；mamba-state sizing 待 linear-attn 布局单独实现
  cfg.num_layer = num_layer;
  cfg.weight_data_type = TYPE_FP16;
  cfg.use_linear_attn = true;
  return cfg;
}

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
}  // namespace

// ============================================================
// MambaBlockCacheLayout
// ============================================================

TEST(MambaBlockCacheLayoutTest, StoresBlockSize) {
  MambaBlockCacheLayout layout(/*block_cache_size=*/588);

  EXPECT_EQ(layout.GetBlockCacheSize(), 588u);
}

TEST(MambaBlockCacheLayoutTest, ImplementsDedicatedBlockCacheLayoutContract) {
  std::shared_ptr<BaseBlockCacheLayout> layout = std::make_shared<MambaBlockCacheLayout>(/*block_cache_size=*/256);

  EXPECT_EQ(layout->GetBlockCacheSize(), 256u);
}

// ============================================================
// MambaModelCacheLayout via CacheLayoutFactory
// ============================================================

TEST(MambaModelCacheLayoutTest, MlaLinearHybridRegistersMlaKvCache) {
  // MLA + LA 混合模型（如 Kimi）：kv-cache 走 Mla 布局而非假定 MHA。
  auto model_cfg = MakeMambaModelConfig(/*num_layer=*/10);
  model_cfg.use_mla = true;
  const auto pipeline_cfg = MakePipelineConfig(model_cfg.num_layer);
  const auto runtime_cfg = MakeRuntimeConfig();

  auto layout = CacheLayoutFactory::CreateModelCacheLayout(model_cfg, pipeline_cfg, runtime_cfg);
  ASSERT_TRUE(layout->IsBlockCacheTypeExist("kv-cache"));
  ASSERT_TRUE(layout->IsBlockCacheTypeExist("mamba-state"));
  EXPECT_NE(std::dynamic_pointer_cast<MlaBlockCacheLayout>(layout->GetBlockCacheLayout("kv-cache")), nullptr);
}

TEST(MambaModelCacheLayoutTest, NoConfigOverloadRegistersFlagOnly) {
  // The inference-executor path creates the layout without configs; it must
  // still report both cache types so the mamba allocator manager is created.
  const auto model_cfg = MakeMambaModelConfig(/*num_layer=*/10);
  auto layout = CacheLayoutFactory::CreateModelCacheLayout(model_cfg);
  ASSERT_TRUE(layout->IsBlockCacheTypeExist("kv-cache"));
  ASSERT_TRUE(layout->IsBlockCacheTypeExist("mamba-state"));
  auto mamba_layout = std::dynamic_pointer_cast<MambaBlockCacheLayout>(layout->GetBlockCacheLayout("mamba-state"));
  ASSERT_NE(mamba_layout, nullptr);
  EXPECT_EQ(mamba_layout->GetBlockCacheSize(), 0u);
}

TEST(MambaModelCacheLayoutTest, NonHybridModelsUnchanged) {
  // Non-hybrid models still get the plain MHA layout without mamba-state.
  ModelConfig model_cfg;
  model_cfg.num_layer = 4;
  model_cfg.num_key_value_heads = 8;
  model_cfg.size_per_head = 128;
  model_cfg.weight_data_type = TYPE_FP16;
  model_cfg.use_linear_attn = false;

  auto layout = CacheLayoutFactory::CreateModelCacheLayout(model_cfg);
  EXPECT_FALSE(layout->IsBlockCacheTypeExist("mamba-state"));
  EXPECT_NE(layout->GetBlockCacheLayout("kv-cache"), nullptr);
}
}  // namespace ksana_llm
