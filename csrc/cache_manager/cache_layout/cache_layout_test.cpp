/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache_manager/cache_layout/cache_layout_factory.h"
#include "cache_manager/cache_layout/deepseek_v4_cache_layout.h"
#include "cache_manager/cache_layout/dsa_cache_layout.h"
#include "cache_manager/cache_layout/mha_cache_layout.h"
#include "cache_manager/cache_layout/mla_cache_layout.h"
#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "device/device_types.h"

namespace ksana_llm {

// ============================================================
// Helper: build minimal configs for each layout type
// ============================================================

// Build a PipelineConfig that covers [0, num_layer-1] with no MTP layers.
static PipelineConfig MakePipelineConfig(uint32_t num_layer) {
  PipelineConfig cfg;
  cfg.lower_layer_idx = 0;
  cfg.upper_layer_idx = static_cast<int16_t>(num_layer - 1);
  // Disable MTP: nextn indices stay at -1 (< num_layer), so predict_nextn == false.
  cfg.lower_nextn_layer_idx = -1;
  cfg.upper_nextn_layer_idx = -1;
  return cfg;
}

// Build a PipelineConfig that covers [0, num_layer-1] with MTP layers appended.
static PipelineConfig MakePipelineConfigWithMtp(uint32_t num_layer, uint32_t nextn_layers) {
  PipelineConfig cfg;
  cfg.lower_layer_idx = 0;
  cfg.upper_layer_idx = static_cast<int16_t>(num_layer - 1);
  // MTP: lower_nextn_layer_idx >= num_layer triggers predict_nextn == true.
  cfg.lower_nextn_layer_idx = static_cast<int16_t>(num_layer);
  cfg.upper_nextn_layer_idx = static_cast<int16_t>(num_layer + nextn_layers - 1);
  return cfg;
}

// Build a RuntimeConfig for MHA (standard attention).
static RuntimeConfig MakeMhaRuntimeConfig(size_t block_token_num = 16, size_t tp_size = 1,
                                          DataType kv_dtype = TYPE_FP16) {
  RuntimeConfig cfg;
  cfg.attn_backend_config.block_token_num = block_token_num;
  cfg.attn_backend_config.kv_cache_dtype = kv_dtype;
  cfg.parallel_basic_config.attn_tensor_parallel_size = tp_size;
  return cfg;
}

// Build a RuntimeConfig for MLA / DSA layouts.
static RuntimeConfig MakeMlaRuntimeConfig(size_t block_token_num = 16, DataType kv_dtype = TYPE_FP16) {
  RuntimeConfig cfg;
  cfg.attn_backend_config.block_token_num = block_token_num;
  cfg.attn_backend_config.kv_cache_dtype = kv_dtype;
  cfg.parallel_basic_config.attn_tensor_parallel_size = 1;
  return cfg;
}

// Build a minimal MHA ModelConfig.
static ModelConfig MakeMhaModelConfig(uint32_t num_layer = 4, uint32_t num_kv_heads = 8, uint32_t size_per_head = 128) {
  ModelConfig cfg;
  cfg.num_layer = num_layer;
  cfg.num_key_value_heads = num_kv_heads;
  cfg.size_per_head = size_per_head;
  cfg.weight_data_type = TYPE_FP16;
  cfg.use_mla = false;
  cfg.use_dsa = false;
  cfg.use_dsv4_attn = false;
  return cfg;
}

// Build a minimal MLA ModelConfig.
static ModelConfig MakeMlaModelConfig(uint32_t num_layer = 4, uint32_t kv_lora_rank = 512,
                                      uint32_t qk_rope_head_dim = 64) {
  ModelConfig cfg;
  cfg.num_layer = num_layer;
  cfg.weight_data_type = TYPE_FP16;
  cfg.use_mla = true;
  cfg.use_dsa = false;
  cfg.use_dsv4_attn = false;
  cfg.mla_config.kv_lora_rank = kv_lora_rank;
  cfg.mla_config.qk_rope_head_dim = qk_rope_head_dim;
  return cfg;
}

// Build a minimal DSA ModelConfig.
static ModelConfig MakeDsaModelConfig(uint32_t num_layer = 4, uint32_t kv_lora_rank = 512,
                                      uint32_t qk_rope_head_dim = 64, uint32_t index_head_dim = 256) {
  ModelConfig cfg;
  cfg.num_layer = num_layer;
  cfg.weight_data_type = TYPE_FP16;
  cfg.use_mla = true;
  cfg.use_dsa = true;
  cfg.use_dsv4_attn = false;
  cfg.mla_config.kv_lora_rank = kv_lora_rank;
  cfg.mla_config.qk_rope_head_dim = qk_rope_head_dim;
  cfg.dsa_config.index_head_dim = index_head_dim;
  return cfg;
}

// Build a minimal DeepSeekV4 ModelConfig.
// compress_ratios: one entry per layer (0 = no cache, 4 = indexer+attn, 128 = attn only).
static ModelConfig MakeDsv4ModelConfig(uint32_t num_layer, const std::vector<uint32_t>& compress_ratios,
                                       uint32_t kv_lora_rank = 512, uint32_t qk_rope_head_dim = 64,
                                       uint32_t index_head_dim = 256) {
  ModelConfig cfg;
  cfg.num_layer = num_layer;
  cfg.weight_data_type = TYPE_FP16;
  cfg.size_per_head = kv_lora_rank + qk_rope_head_dim;
  cfg.use_mla = true;
  cfg.use_dsa = true;
  cfg.use_dsv4_attn = true;
  cfg.mla_config.kv_lora_rank = kv_lora_rank;
  cfg.mla_config.qk_rope_head_dim = qk_rope_head_dim;
  cfg.dsa_config.index_head_dim = index_head_dim;
  cfg.dsv4_attn_config.compress_ratios = compress_ratios;
  return cfg;
}

// ============================================================
// DsaBlockCacheLayout tests
// ============================================================

class DsaCacheLayoutTest : public ::testing::Test {};

TEST_F(DsaCacheLayoutTest, BlockCacheSizeIsPositive) {
  const uint32_t num_layer = 4;
  auto model_cfg = MakeDsaModelConfig(num_layer, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DsaBlockCacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);
  EXPECT_GT(layout.GetBlockCacheSize(), 0u);
}

TEST_F(DsaCacheLayoutTest, BlockCacheSizeMatchesManualCalculation) {
  // Manual formula (non-fp8 MLA path):
  //   indexer_key_bytes  = index_head_dim * sizeof(fp8_e4m3)
  //   indexer_val_bytes  = index_head_dim / 128 * sizeof(fp32)
  //   indexer_per_layer  = (indexer_key_bytes + indexer_val_bytes) * block_token_num
  //   mla_per_token      = (kv_lora_rank + qk_rope_head_dim) * sizeof(fp16)
  //   mla_per_layer      = mla_per_token * block_token_num
  //   total              = (indexer_per_layer + mla_per_layer) * num_layer
  const uint32_t num_layer = 4;
  const uint32_t kv_lora_rank = 512;
  const uint32_t qk_rope_head_dim = 64;
  const uint32_t index_head_dim = 256;
  const size_t block_token_num = 16;
  constexpr size_t QUANT_BLOCK_SIZE = 128;

  auto model_cfg = MakeDsaModelConfig(num_layer, kv_lora_rank, qk_rope_head_dim, index_head_dim);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DsaBlockCacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);

  const size_t indexer_key_bytes = index_head_dim * GetTypeSize(TYPE_FP8_E4M3);
  const size_t indexer_val_bytes = index_head_dim / QUANT_BLOCK_SIZE * GetTypeSize(TYPE_FP32);
  const size_t indexer_per_layer = (indexer_key_bytes + indexer_val_bytes) * block_token_num;

  const size_t mla_per_token = (kv_lora_rank + qk_rope_head_dim) * GetTypeSize(TYPE_FP16);
  const size_t mla_per_layer = mla_per_token * block_token_num;

  const size_t expected = (indexer_per_layer + mla_per_layer) * num_layer;
  EXPECT_EQ(layout.GetBlockCacheSize(), expected);
}

TEST_F(DsaCacheLayoutTest, IndexerAndAttentionOffsetsAreOrdered) {
  // indexer occupies the lower part of each layer's region;
  // attention starts right after the indexer.
  const uint32_t num_layer = 4;
  auto model_cfg = MakeDsaModelConfig(num_layer, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DsaBlockCacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);
  for (int32_t i = 0; i < static_cast<int32_t>(num_layer); ++i) {
    auto idx_offsets = layout.GetKvCacheLayerOffsets(i, "indexer");
    auto attn_offsets = layout.GetKvCacheLayerOffsets(i, "attention");

    // Indexer K comes before attention K.
    EXPECT_LE(idx_offsets.k_offset, attn_offsets.k_offset)
        << "indexer k_offset must be <= attention k_offset at layer " << i;
    // Indexer V (scale) is between indexer K and attention K.
    EXPECT_GE(idx_offsets.v_offset, idx_offsets.k_offset)
        << "indexer v_offset must be >= indexer k_offset at layer " << i;
    EXPECT_LE(idx_offsets.v_offset, attn_offsets.k_offset)
        << "indexer v_offset must be <= attention k_offset at layer " << i;
    // Attention K and V share the same offset (MLA property).
    EXPECT_EQ(attn_offsets.k_offset, attn_offsets.v_offset) << "DSA attention K and V must share offset at layer " << i;
  }
}

TEST_F(DsaCacheLayoutTest, Layer0IndexerKOffsetStartsAtZero) {
  const uint32_t num_layer = 4;
  auto model_cfg = MakeDsaModelConfig(num_layer, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DsaBlockCacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);
  auto offsets = layout.GetKvCacheLayerOffsets(0, "indexer");
  EXPECT_EQ(offsets.k_offset, 0u);
}

TEST_F(DsaCacheLayoutTest, GetActiveKvCacheTypesReturnsIndexerAndAttention) {
  auto model_cfg = MakeDsaModelConfig(2, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(2);
  auto runtime_cfg = MakeMlaRuntimeConfig(8, TYPE_FP16);

  DsaBlockCacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);
  auto types = layout.GetActiveKvCacheTypes();
  ASSERT_EQ(types.size(), 2u);
  EXPECT_EQ(types[0], "indexer");
  EXPECT_EQ(types[1], "attention");
}

TEST_F(DsaCacheLayoutTest, OutOfRangeLayerIndexThrows) {
  const uint32_t num_layer = 2;
  auto model_cfg = MakeDsaModelConfig(num_layer, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(8, TYPE_FP16);

  DsaBlockCacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(static_cast<int32_t>(num_layer), "attention"), std::runtime_error);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(-1, "indexer"), std::runtime_error);
}

TEST_F(DsaCacheLayoutTest, LayerOffsetsAreContiguous) {
  // The attention offset of layer i+1 should equal the base offset of layer i+1,
  // which is the sum of all previous layers' cache sizes.
  const uint32_t num_layer = 4;
  const uint32_t kv_lora_rank = 512;
  const uint32_t qk_rope_head_dim = 64;
  const uint32_t index_head_dim = 256;
  const size_t block_token_num = 16;
  constexpr size_t QUANT_BLOCK_SIZE = 128;

  auto model_cfg = MakeDsaModelConfig(num_layer, kv_lora_rank, qk_rope_head_dim, index_head_dim);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DsaBlockCacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);

  const size_t indexer_key_bytes = index_head_dim * GetTypeSize(TYPE_FP8_E4M3);
  const size_t indexer_val_bytes = index_head_dim / QUANT_BLOCK_SIZE * GetTypeSize(TYPE_FP32);
  const size_t indexer_per_layer = (indexer_key_bytes + indexer_val_bytes) * block_token_num;
  const size_t mla_per_layer = (kv_lora_rank + qk_rope_head_dim) * GetTypeSize(TYPE_FP16) * block_token_num;
  const size_t per_layer = indexer_per_layer + mla_per_layer;

  for (int32_t i = 1; i < static_cast<int32_t>(num_layer); ++i) {
    auto idx_offsets = layout.GetKvCacheLayerOffsets(i, "indexer");
    EXPECT_EQ(idx_offsets.k_offset, static_cast<size_t>(i) * per_layer)
        << "Layer " << i << " indexer k_offset should be i * per_layer";
  }
}

// ============================================================
// DeepSeekV4CacheLayout tests (主 KV layout 只含 indexer + attention)
// ============================================================

// c4 compressor overlap tail 由 DeepSeekV4SwaWithTailCacheLayout (block_cache_type = "SWA-cache") 管理;
// 主 KV layout 不持有 swa / tail_attn / tail_indexer。下列 helper 与
// deepseek_v4_cache_layout.h 中的常量保持一致, 供主 KV / SWA 两套测试复用。
//   kTailTokens  = 4   (一个 c4 group 的 token 数)
//   kCoffMax     = 2   (c4 overlap window 系数)
//   kKvScorePair = 2   (kv FP32 + score FP32)
inline size_t ExpectedTailAttnSize(uint32_t size_per_head) {
  return 4u * 2u * 2u * size_per_head * GetTypeSize(TYPE_FP32);  // 64 * size_per_head bytes
}
inline size_t ExpectedTailIndexerSize(uint32_t index_head_dim) {
  return 4u * 2u * 2u * index_head_dim * GetTypeSize(TYPE_FP32);  // 64 * index_head_dim bytes
}
inline size_t ExpectedTailPerLayer(uint32_t size_per_head, uint32_t index_head_dim) {
  return ExpectedTailAttnSize(size_per_head) + ExpectedTailIndexerSize(index_head_dim);
}

class DeepSeekV4CacheLayoutTest : public ::testing::Test {};

TEST_F(DeepSeekV4CacheLayoutTest, AllZeroCompressRatiosGivesZeroBlockSize) {
  // compress_ratio == 0 for every layer means no indexer/attention cache.
  // swa/tail 不贡献到主 KV layout, 总大小为 0。
  const uint32_t num_layer = 4;
  std::vector<uint32_t> ratios(num_layer, 0u);
  const uint32_t kv_lora_rank = 512;
  const uint32_t qk_rope_head_dim = 64;
  const uint32_t index_head_dim = 256;
  const size_t block_token_num = 16;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, kv_lora_rank, qk_rope_head_dim, index_head_dim);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4CacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);
  EXPECT_EQ(layout.GetBlockCacheSize(), 0u);
}

TEST_F(DeepSeekV4CacheLayoutTest, CompressRatio128NoIndexerCache) {
  // compress_ratio == 128: no indexer; attention cache = DSA size / 128; 主 KV layout 不含 SWA/tail。
  const uint32_t num_layer = 2;
  std::vector<uint32_t> ratios(num_layer, 128u);
  const uint32_t kv_lora_rank = 512;
  const uint32_t qk_rope_head_dim = 64;
  const uint32_t index_head_dim = 256;
  const size_t block_token_num = 128;  // must be divisible by compress_ratio

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, kv_lora_rank, qk_rope_head_dim, index_head_dim);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4CacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);

  const size_t size_per_head = kv_lora_rank + qk_rope_head_dim;
  const size_t mla_per_token = size_per_head * GetTypeSize(TYPE_FP16);
  const size_t attn_per_layer = mla_per_token * (block_token_num / 128);
  EXPECT_EQ(layout.GetBlockCacheSize(), attn_per_layer * num_layer);
}

TEST_F(DeepSeekV4CacheLayoutTest, CompressRatio4HasIndexerAndAttention) {
  // compress_ratio == 4: indexer present; attention cache = DSA size / 4; 主 KV layout 不含 SWA/tail。
  const uint32_t num_layer = 2;
  std::vector<uint32_t> ratios(num_layer, 4u);
  const uint32_t kv_lora_rank = 512;
  const uint32_t qk_rope_head_dim = 64;
  const uint32_t index_head_dim = 256;
  const size_t block_token_num = 16;
  constexpr size_t QUANT_BLOCK_SIZE = 128;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, kv_lora_rank, qk_rope_head_dim, index_head_dim);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4CacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);

  const size_t compressed_block_token_num = block_token_num / 4;
  const size_t indexer_key_bytes = index_head_dim * GetTypeSize(TYPE_FP8_E4M3);
  const size_t indexer_val_bytes = index_head_dim / QUANT_BLOCK_SIZE * GetTypeSize(TYPE_FP32);
  const size_t indexer_per_layer = (indexer_key_bytes + indexer_val_bytes) * compressed_block_token_num;

  const size_t size_per_head = kv_lora_rank + qk_rope_head_dim;
  const size_t mla_per_token = size_per_head * GetTypeSize(TYPE_FP16);
  const size_t attn_per_layer = mla_per_token * compressed_block_token_num;

  EXPECT_EQ(layout.GetBlockCacheSize(), (indexer_per_layer + attn_per_layer) * num_layer);
}

TEST_F(DeepSeekV4CacheLayoutTest, MixedCompressRatiosBlockSizeIsSum) {
  // Layer 0: ratio=0 (no cache), Layer 1: ratio=128, Layer 2: ratio=4, Layer 3: ratio=0
  // 主 KV layout 仅 indexer + attention; swa/tail 由 SWA-cache layout 管理。
  const uint32_t num_layer = 4;
  std::vector<uint32_t> ratios = {0u, 128u, 4u, 0u};
  const uint32_t kv_lora_rank = 512;
  const uint32_t qk_rope_head_dim = 64;
  const uint32_t index_head_dim = 256;
  const size_t block_token_num = 128;
  constexpr size_t QUANT_BLOCK_SIZE = 128;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, kv_lora_rank, qk_rope_head_dim, index_head_dim);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4CacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);

  // ---- Pre-compute per-layer sizes ----
  const size_t mla_per_token = (kv_lora_rank + qk_rope_head_dim) * GetTypeSize(TYPE_FP16);

  // Layer 0 (ratio=0): cache_size = 0
  const size_t layer0_base = 0;
  const size_t layer0_size = 0;
  // Layer 1 (ratio=128): attn = mla_per_token * (block_token_num / 128)
  const size_t layer1_base = layer0_base + layer0_size;
  const size_t layer1_attn = mla_per_token * (block_token_num / 128);
  const size_t layer1_size = layer1_attn;
  // Layer 2 (ratio=4): indexer + compressed attention
  const size_t layer2_base = layer1_base + layer1_size;
  const size_t compressed_block_token_num_4 = block_token_num / 4;
  const size_t indexer_key_bytes = index_head_dim * GetTypeSize(TYPE_FP8_E4M3);
  const size_t indexer_val_bytes = index_head_dim / QUANT_BLOCK_SIZE * GetTypeSize(TYPE_FP32);
  const size_t indexer_size = (indexer_key_bytes + indexer_val_bytes) * compressed_block_token_num_4;
  const size_t layer2_attn = mla_per_token * compressed_block_token_num_4;
  const size_t layer2_size = indexer_size + layer2_attn;
  // Layer 3 (ratio=0): cache_size = 0
  const size_t layer3_base = layer2_base + layer2_size;
  const size_t layer3_size = 0;

  EXPECT_EQ(layout.GetBlockCacheSize(), layer0_size + layer1_size + layer2_size + layer3_size);

  // ---- Verify per-layer GetKvCacheLayerOffsets ----
  // Layer 0 (ratio=0): indexer/attention offsets == base.
  {
    auto idx = layout.GetKvCacheLayerOffsets(0, "indexer");
    auto attn = layout.GetKvCacheLayerOffsets(0, "attention");
    EXPECT_EQ(idx.k_offset, layer0_base);
    EXPECT_EQ(idx.v_offset, layer0_base);
    EXPECT_EQ(attn.k_offset, layer0_base);
    EXPECT_EQ(attn.v_offset, layer0_base);
  }
  // Layer 1 (ratio=128): no indexer → indexer offsets == layer base.
  {
    auto idx = layout.GetKvCacheLayerOffsets(1, "indexer");
    auto attn = layout.GetKvCacheLayerOffsets(1, "attention");
    EXPECT_EQ(idx.k_offset, layer1_base);
    EXPECT_EQ(idx.v_offset, layer1_base);
    EXPECT_EQ(attn.k_offset, layer1_base);
    EXPECT_EQ(attn.v_offset, layer1_base);
  }
  // Layer 2 (ratio=4): indexer present, then attention.
  {
    auto idx = layout.GetKvCacheLayerOffsets(2, "indexer");
    auto attn = layout.GetKvCacheLayerOffsets(2, "attention");
    EXPECT_EQ(idx.k_offset, layer2_base);
    EXPECT_EQ(idx.v_offset, layer2_base + indexer_key_bytes * compressed_block_token_num_4);
    EXPECT_EQ(attn.k_offset, layer2_base + indexer_size);
    EXPECT_EQ(attn.v_offset, attn.k_offset);
  }
  // Layer 3 (ratio=0): indexer/attention offsets == base.
  {
    auto idx = layout.GetKvCacheLayerOffsets(3, "indexer");
    auto attn = layout.GetKvCacheLayerOffsets(3, "attention");
    EXPECT_EQ(idx.k_offset, layer3_base);
    EXPECT_EQ(idx.v_offset, layer3_base);
    EXPECT_EQ(attn.k_offset, layer3_base);
    EXPECT_EQ(attn.v_offset, layer3_base);
  }
  // 主 KV layout 不持有 swa/tail, 查询时必须 throw (这些类型已迁到 SWA-cache layout)。
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(2, "swa"), std::runtime_error);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(2, "tail_attn"), std::runtime_error);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(2, "tail_indexer"), std::runtime_error);
}

TEST_F(DeepSeekV4CacheLayoutTest, ZeroRatioLayerOffsetsAllPointToBase) {
  // For compress_ratio == 0, indexer/attention offsets all point to base; swa 由 SWA-cache layout 管理。
  const uint32_t num_layer = 2;
  std::vector<uint32_t> ratios = {0u, 128u};
  const size_t block_token_num = 128;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4CacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);

  auto idx_offsets = layout.GetKvCacheLayerOffsets(0, "indexer");
  auto attn_offsets = layout.GetKvCacheLayerOffsets(0, "attention");
  EXPECT_EQ(idx_offsets.k_offset, 0u);
  EXPECT_EQ(idx_offsets.v_offset, 0u);
  EXPECT_EQ(attn_offsets.k_offset, 0u);
  EXPECT_EQ(attn_offsets.v_offset, 0u);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(0, "swa"), std::runtime_error);
}

TEST_F(DeepSeekV4CacheLayoutTest, Ratio128LayerIndexerOffsetsEqualBase) {
  // For compress_ratio == 128, indexer offsets should point to the layer base
  // (no indexer cache allocated).
  const uint32_t num_layer = 1;
  std::vector<uint32_t> ratios = {128u};
  const size_t block_token_num = 128;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4CacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);

  auto idx_offsets = layout.GetKvCacheLayerOffsets(0, "indexer");
  auto attn_offsets = layout.GetKvCacheLayerOffsets(0, "attention");

  // No indexer: both indexer offsets equal the layer base (0 for layer 0).
  EXPECT_EQ(idx_offsets.k_offset, 0u);
  EXPECT_EQ(idx_offsets.v_offset, 0u);
  // Attention starts at base (no indexer prefix).
  EXPECT_EQ(attn_offsets.k_offset, 0u);
  EXPECT_EQ(attn_offsets.v_offset, attn_offsets.k_offset);
}

TEST_F(DeepSeekV4CacheLayoutTest, Ratio4LayerAttentionStartsAfterIndexer) {
  // For compress_ratio == 4, attention k_offset must be > indexer k_offset.
  const uint32_t num_layer = 1;
  std::vector<uint32_t> ratios = {4u};
  const size_t block_token_num = 16;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4CacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);

  auto idx_offsets = layout.GetKvCacheLayerOffsets(0, "indexer");
  auto attn_offsets = layout.GetKvCacheLayerOffsets(0, "attention");

  EXPECT_LT(idx_offsets.k_offset, attn_offsets.k_offset);
  EXPECT_EQ(attn_offsets.k_offset, attn_offsets.v_offset);
}

TEST_F(DeepSeekV4CacheLayoutTest, GetActiveKvCacheTypesReturnsIndexerAndAttention) {
  std::vector<uint32_t> ratios = {4u, 128u};
  auto model_cfg = MakeDsv4ModelConfig(2, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(2);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DeepSeekV4CacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);
  auto types = layout.GetActiveKvCacheTypes();
  ASSERT_EQ(types.size(), 2u);
  EXPECT_EQ(types[0], "indexer");
  EXPECT_EQ(types[1], "attention");
}

TEST_F(DeepSeekV4CacheLayoutTest, OutOfRangeLayerIndexThrows) {
  const uint32_t num_layer = 2;
  std::vector<uint32_t> ratios(num_layer, 128u);
  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(128, TYPE_FP16);

  DeepSeekV4CacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(static_cast<int32_t>(num_layer), "attention"), std::runtime_error);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(-1, "attention"), std::runtime_error);
}

TEST_F(DeepSeekV4CacheLayoutTest, MtpLayersContributeNothingToMainKvLayout) {
  // MTP 层 (ratio=0) 在主 KV layout 中 cache_size 为 0, 启用 MTP 不会增大主 KV pool 单 block 大小。
  // 注: SWA pool (DeepSeekV4SwaWithTailCacheLayout) 仍会为 MTP 层分配 swa+tail,
  // 该部分覆盖见 DeepSeekV4SwaWithTailCacheLayoutTest.MtpLayersAddSwaAndTail。
  const uint32_t num_layer = 2;
  const uint32_t nextn_layers = 2;
  const size_t block_token_num = 128;
  std::vector<uint32_t> ratios = {128u, 128u, 0u, 0u};

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  model_cfg.num_nextn_predict_layers = nextn_layers;

  auto pipeline_no_mtp = MakePipelineConfig(num_layer);
  auto pipeline_with_mtp = MakePipelineConfigWithMtp(num_layer, nextn_layers);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4CacheLayout layout_no_mtp(model_cfg, pipeline_no_mtp, runtime_cfg);
  DeepSeekV4CacheLayout layout_with_mtp(model_cfg, pipeline_with_mtp, runtime_cfg);

  // MTP 层 ratio=0 → cache_size 0 → 启用 MTP 不增大主 KV layout 总大小。
  EXPECT_EQ(layout_with_mtp.GetBlockCacheSize(), layout_no_mtp.GetBlockCacheSize());
}

// ============================================================
// DeepSeekV4SwaWithTailCacheLayout tests (SWA pool 专属 layout)
// ============================================================

class DeepSeekV4SwaWithTailCacheLayoutTest : public ::testing::Test {};

TEST_F(DeepSeekV4SwaWithTailCacheLayoutTest, BlockCacheSizeFormulaIsLayerNumTimesPerLayer) {
  // 每层贡献 swa_cache_size + tail_attn_size + tail_indexer_size，公式与 compress_ratio 无关。
  const uint32_t num_layer = 3;
  std::vector<uint32_t> ratios = {0u, 128u, 4u};
  const uint32_t kv_lora_rank = 512;
  const uint32_t qk_rope_head_dim = 64;
  const uint32_t index_head_dim = 256;
  const size_t block_token_num = 128;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, kv_lora_rank, qk_rope_head_dim, index_head_dim);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4SwaWithTailCacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);

  const size_t size_per_head = kv_lora_rank + qk_rope_head_dim;
  const size_t swa_per_layer = size_per_head * GetTypeSize(TYPE_FP16) * block_token_num;
  const size_t tail_per_layer = ExpectedTailPerLayer(size_per_head, index_head_dim);
  EXPECT_EQ(layout.GetBlockCacheSize(), (swa_per_layer + tail_per_layer) * num_layer);
}

TEST_F(DeepSeekV4SwaWithTailCacheLayoutTest, PerLayerOffsetsAreContiguous) {
  // 每层 layout: [swa | tail_attn | tail_indexer], offset 顺序严格递增。
  const uint32_t num_layer = 3;
  std::vector<uint32_t> ratios = {0u, 128u, 4u};
  const uint32_t kv_lora_rank = 512;
  const uint32_t qk_rope_head_dim = 64;
  const uint32_t index_head_dim = 256;
  const size_t block_token_num = 128;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, kv_lora_rank, qk_rope_head_dim, index_head_dim);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4SwaWithTailCacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);

  const size_t size_per_head = kv_lora_rank + qk_rope_head_dim;
  const size_t swa_size = size_per_head * GetTypeSize(TYPE_FP16) * block_token_num;
  const size_t tail_attn_size = ExpectedTailAttnSize(size_per_head);
  const size_t tail_idx_size = ExpectedTailIndexerSize(index_head_dim);
  const size_t per_layer = swa_size + tail_attn_size + tail_idx_size;

  for (int32_t i = 0; i < static_cast<int32_t>(num_layer); ++i) {
    const size_t layer_base = static_cast<size_t>(i) * per_layer;
    auto swa = layout.GetKvCacheLayerOffsets(i, "swa");
    auto tail_attn = layout.GetKvCacheLayerOffsets(i, "tail_attn");
    auto tail_idx = layout.GetKvCacheLayerOffsets(i, "tail_indexer");
    EXPECT_EQ(swa.k_offset, layer_base) << "layer " << i;
    EXPECT_EQ(swa.v_offset, swa.k_offset) << "layer " << i;
    EXPECT_EQ(tail_attn.k_offset, layer_base + swa_size) << "layer " << i;
    EXPECT_EQ(tail_attn.v_offset, tail_attn.k_offset) << "layer " << i;
    EXPECT_EQ(tail_idx.k_offset, layer_base + swa_size + tail_attn_size) << "layer " << i;
    EXPECT_EQ(tail_idx.v_offset, tail_idx.k_offset) << "layer " << i;
  }
}

TEST_F(DeepSeekV4SwaWithTailCacheLayoutTest, MtpLayersAddSwaAndTail) {
  // 启用 MTP 后 SWA pool 应额外为 MTP 层分配 swa+tail (与主 KV layout 不同, 这里 ratio 不影响 swa+tail 分配)。
  const uint32_t num_layer = 2;
  const uint32_t nextn_layers = 2;
  const uint32_t kv_lora_rank = 512;
  const uint32_t qk_rope_head_dim = 64;
  const uint32_t index_head_dim = 256;
  const size_t block_token_num = 128;
  std::vector<uint32_t> ratios = {128u, 128u, 0u, 0u};

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, kv_lora_rank, qk_rope_head_dim, index_head_dim);
  model_cfg.num_nextn_predict_layers = nextn_layers;

  auto pipeline_no_mtp = MakePipelineConfig(num_layer);
  auto pipeline_with_mtp = MakePipelineConfigWithMtp(num_layer, nextn_layers);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4SwaWithTailCacheLayout layout_no_mtp(model_cfg, pipeline_no_mtp, runtime_cfg);
  DeepSeekV4SwaWithTailCacheLayout layout_with_mtp(model_cfg, pipeline_with_mtp, runtime_cfg);

  const size_t size_per_head = kv_lora_rank + qk_rope_head_dim;
  const size_t swa_per_layer = size_per_head * GetTypeSize(TYPE_FP16) * block_token_num;
  const size_t tail_per_layer = ExpectedTailPerLayer(size_per_head, index_head_dim);
  const size_t expected_diff = (swa_per_layer + tail_per_layer) * nextn_layers;
  EXPECT_EQ(layout_with_mtp.GetBlockCacheSize(), layout_no_mtp.GetBlockCacheSize() + expected_diff);
}

TEST_F(DeepSeekV4SwaWithTailCacheLayoutTest, GetActiveKvCacheTypesIsEmpty) {
  // SWA pool 不参与 model_input.cpp 的 PrepareKVCacheBlocks 循环 (Dsv4MqaMeta 自己处理),
  // GetActiveKvCacheTypes 必须返回空。
  std::vector<uint32_t> ratios = {4u, 128u};
  auto model_cfg = MakeDsv4ModelConfig(2, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(2);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);
  DeepSeekV4SwaWithTailCacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);
  EXPECT_TRUE(layout.GetActiveKvCacheTypes().empty());
}

TEST_F(DeepSeekV4SwaWithTailCacheLayoutTest, UnsupportedKvCacheTypeOrIndexThrows) {
  std::vector<uint32_t> ratios = {4u, 128u};
  auto model_cfg = MakeDsv4ModelConfig(2, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(2);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);
  DeepSeekV4SwaWithTailCacheLayout layout(model_cfg, pipeline_cfg, runtime_cfg);

  // 不支持的 kv_cache_type 必须 throw (避免静默走错 layout)。
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(0, "indexer"), std::runtime_error);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(0, "attention"), std::runtime_error);
  // layer_index 越界必须 throw。
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(2, "swa"), std::runtime_error);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(-1, "swa"), std::runtime_error);
}

// ============================================================
// DeepSeekV4CompressorStateLayout tests
// ============================================================

class DeepSeekV4CompressorStateLayoutTest : public ::testing::Test {};

TEST_F(DeepSeekV4CompressorStateLayoutTest, BlockCompressRatio0GivesZeroForRatio0Layers) {
  // block_compress_ratio == 0: layers with layer_compress_ratio == 0 match, but their
  // contribution is coff * 0 * 2 * head_dim * sizeof(fp32) = 0.
  const uint32_t num_layer = 4;
  std::vector<uint32_t> ratios = {0u, 4u, 128u, 0u};

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, /*block_compress_ratio=*/0u, false);

  // Layers 0 and 3 have ratio==0, match block_compress_ratio==0, but contribute 0 bytes.
  EXPECT_EQ(layout.GetBlockCacheSize(), 0u);
}

TEST_F(DeepSeekV4CompressorStateLayoutTest, BlockCompressRatio4OnlyActivatesRatio4Layers) {
  // block_compress_ratio == 4: only layers with layer_compress_ratio == 4 are allocated.
  const uint32_t num_layer = 4;
  std::vector<uint32_t> ratios = {0u, 4u, 128u, 4u};
  const uint32_t kv_lora_rank = 512;
  const uint32_t qk_rope_head_dim = 64;
  const size_t block_token_num = 16;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, kv_lora_rank, qk_rope_head_dim, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, /*block_compress_ratio=*/4u, false);

  // Layers 1 and 3 have ratio==4; layers 0 and 2 are inactive.
  // coff = 1 + (4==4) = 2; per active layer: 2 * 4 * 2 * head_dim * sizeof(fp32)
  const size_t head_dim = kv_lora_rank + qk_rope_head_dim;
  const size_t per_layer = 2u * 4u * 2u * head_dim * GetTypeSize(TYPE_FP32);
  EXPECT_EQ(layout.GetBlockCacheSize(), per_layer * 2);
}

TEST_F(DeepSeekV4CompressorStateLayoutTest, BlockCompressRatio128OnlyActivatesRatio128Layers) {
  // block_compress_ratio == 128: only layers with layer_compress_ratio == 128 are allocated.
  const uint32_t num_layer = 4;
  std::vector<uint32_t> ratios = {128u, 4u, 128u, 0u};
  const uint32_t kv_lora_rank = 512;
  const uint32_t qk_rope_head_dim = 64;
  const size_t block_token_num = 16;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, kv_lora_rank, qk_rope_head_dim, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, /*block_compress_ratio=*/128u, false);

  // Layers 0 and 2 have ratio==128; layers 1 and 3 are inactive.
  // coff = 1 + (128==4) = 1; per active layer: 1 * 128 * 2 * head_dim * sizeof(fp32)
  const size_t head_dim = kv_lora_rank + qk_rope_head_dim;
  const size_t per_layer = 1u * 128u * 2u * head_dim * GetTypeSize(TYPE_FP32);
  EXPECT_EQ(layout.GetBlockCacheSize(), per_layer * 2);
}

TEST_F(DeepSeekV4CompressorStateLayoutTest, AllInactiveLayersGivesZeroBlockSize) {
  // block_compress_ratio == 4 but no layer has ratio==4: block size must be 0.
  const uint32_t num_layer = 3;
  std::vector<uint32_t> ratios = {0u, 128u, 0u};

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, /*block_compress_ratio=*/4u, false);
  EXPECT_EQ(layout.GetBlockCacheSize(), 0u);
}

TEST_F(DeepSeekV4CompressorStateLayoutTest, KvShareSameOffsetMlaProperty) {
  // K and V must share the same offset (MLA property).
  const uint32_t num_layer = 3;
  std::vector<uint32_t> ratios = {4u, 4u, 128u};

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, /*block_compress_ratio=*/4u, false);

  for (int32_t i = 0; i < static_cast<int32_t>(num_layer); ++i) {
    auto offsets = layout.GetKvCacheLayerOffsets(i, "attention");
    EXPECT_EQ(offsets.k_offset, offsets.v_offset) << "K and V must share offset at layer " << i;
  }
}

TEST_F(DeepSeekV4CompressorStateLayoutTest, InactiveLayerOffsetEqualsActiveLayerBase) {
  // An inactive layer's offset must equal the cumulative base of the preceding active layers.
  const uint32_t num_layer = 4;
  std::vector<uint32_t> ratios = {0u, 4u, 128u, 4u};
  const uint32_t kv_lora_rank = 512;
  const uint32_t qk_rope_head_dim = 64;
  const size_t block_token_num = 16;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, kv_lora_rank, qk_rope_head_dim, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  // block_compress_ratio == 4: layers 1 and 3 are active; layers 0 and 2 are inactive.
  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, /*block_compress_ratio=*/4u, false);

  // coff=2, per active layer: 2 * 4 * 2 * (kv_lora_rank + qk_rope_head_dim) * sizeof(fp32)
  const size_t head_dim = kv_lora_rank + qk_rope_head_dim;
  const size_t per_layer = 2u * 4u * 2u * head_dim * GetTypeSize(TYPE_FP32);

  // Layer 0 (inactive): offset == 0
  EXPECT_EQ(layout.GetKvCacheLayerOffsets(0, "attention").k_offset, 0u);
  // Layer 1 (active): offset == 0 (first active layer)
  EXPECT_EQ(layout.GetKvCacheLayerOffsets(1, "attention").k_offset, 0u);
  // Layer 2 (inactive): offset == per_layer (after layer 1's allocation)
  EXPECT_EQ(layout.GetKvCacheLayerOffsets(2, "attention").k_offset, per_layer);
  // Layer 3 (active): offset == per_layer (same base as layer 2, which is inactive)
  EXPECT_EQ(layout.GetKvCacheLayerOffsets(3, "attention").k_offset, per_layer);
}

TEST_F(DeepSeekV4CompressorStateLayoutTest, GetActiveKvCacheTypesReturnsAttention) {
  std::vector<uint32_t> ratios = {4u, 128u};
  auto model_cfg = MakeDsv4ModelConfig(2, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(2);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, /*block_compress_ratio=*/4u, false);
  auto types = layout.GetActiveKvCacheTypes();
  ASSERT_EQ(types.size(), 1u);
  EXPECT_EQ(types[0], "attention");
}

TEST_F(DeepSeekV4CompressorStateLayoutTest, OutOfRangeLayerIndexThrows) {
  const uint32_t num_layer = 2;
  std::vector<uint32_t> ratios(num_layer, 4u);
  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, /*block_compress_ratio=*/4u, false);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(static_cast<int32_t>(num_layer), "attention"), std::runtime_error);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(-1, "attention"), std::runtime_error);
}

TEST_F(DeepSeekV4CompressorStateLayoutTest, DefaultInitializeUsesRatio0) {
  // The no-arg Initialize() override delegates to block_compress_ratio == 0, is_in_indexer == false.
  const uint32_t num_layer = 3;
  std::vector<uint32_t> ratios = {0u, 4u, 128u};

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  // Construct via the base Initialize() path (block_compress_ratio defaults to 0).
  DeepSeekV4CompressorStateLayout layout;
  layout.Initialize(model_cfg, pipeline_cfg, runtime_cfg);

  // block_compress_ratio=0: only layers with layer_compress_ratio==0 match (layer 0).
  // But contribution is coff * 0 * ... = 0. So total = 0.
  EXPECT_EQ(layout.GetBlockCacheSize(), 0u);
}

TEST_F(DeepSeekV4CompressorStateLayoutTest, MtpLayersBeyondMainLayersAreInactive) {
  // MTP layers have compress_ratio == 0, so with block_compress_ratio == 4
  // those MTP layers are inactive and don't add to block size.
  const uint32_t num_layer = 2;
  const uint32_t nextn_layers = 2;
  // Pad ratios to cover MTP layers (they get ratio=0).
  std::vector<uint32_t> ratios = {4u, 4u, 0u, 0u};

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  model_cfg.num_nextn_predict_layers = nextn_layers;

  auto pipeline_no_mtp = MakePipelineConfig(num_layer);
  auto pipeline_with_mtp = MakePipelineConfigWithMtp(num_layer, nextn_layers);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout_no_mtp(model_cfg, pipeline_no_mtp, runtime_cfg, /*block_compress_ratio=*/4u,
                                                false);
  DeepSeekV4CompressorStateLayout layout_with_mtp(model_cfg, pipeline_with_mtp, runtime_cfg,
                                                  /*block_compress_ratio=*/4u, false);

  // MTP layers have ratio==0, so they are inactive for block_compress_ratio==4.
  EXPECT_EQ(layout_no_mtp.GetBlockCacheSize(), layout_with_mtp.GetBlockCacheSize());
}

// ============================================================
// DeepSeekV4CompressorStateLayout (indexer mode) tests
// ============================================================

class DeepSeekV4CompressorIndexerStateLayoutTest : public ::testing::Test {};

TEST_F(DeepSeekV4CompressorIndexerStateLayoutTest, OnlyRatio4LayersAllocateIndexerCache) {
  // Only layers with compress_ratio == 4 should contribute to the block cache size.
  const uint32_t num_layer = 4;
  std::vector<uint32_t> ratios = {0u, 4u, 128u, 4u};
  const uint32_t index_head_dim = 256;
  const size_t block_token_num = 16;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, index_head_dim);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, 4u, true);

  // coff = 2 (since block_compress_ratio==4), compress_ratio=4, head_dim=index_head_dim
  // Per active layer: coff * compress_ratio * 2 * head_dim * sizeof(fp32) = 2 * 4 * 2 * 256 * 4 = 16384
  const size_t per_active_layer = 2u * 4u * 2u * index_head_dim * GetTypeSize(TYPE_FP32);
  // Layers 1 and 3 are active (ratio==4), so total = per_active_layer * 2
  EXPECT_EQ(layout.GetBlockCacheSize(), per_active_layer * 2);
}

TEST_F(DeepSeekV4CompressorIndexerStateLayoutTest, NoRatio4LayersGivesZeroBlockSize) {
  // No layer has compress_ratio == 4: block size must be 0.
  const uint32_t num_layer = 3;
  std::vector<uint32_t> ratios = {0u, 128u, 0u};

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, 4u, true);
  EXPECT_EQ(layout.GetBlockCacheSize(), 0u);
}

TEST_F(DeepSeekV4CompressorIndexerStateLayoutTest, KvOffsetsShareSameOffset) {
  // For indexer mode (MLA property), k_offset == v_offset for all layers.
  const uint32_t num_layer = 1;
  std::vector<uint32_t> ratios = {4u};
  const uint32_t index_head_dim = 256;
  const size_t block_token_num = 16;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, index_head_dim);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, 4u, true);

  auto offsets = layout.GetKvCacheLayerOffsets(0, "attention");
  EXPECT_EQ(offsets.k_offset, offsets.v_offset) << "MLA property: key and value share the same offset";
}

TEST_F(DeepSeekV4CompressorIndexerStateLayoutTest, InactiveLayerKvOffsetsAreEqual) {
  // For an inactive layer (ratio != 4), k_offset == v_offset (no allocation).
  const uint32_t num_layer = 2;
  std::vector<uint32_t> ratios = {128u, 4u};

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, 4u, true);

  // Layer 0 (ratio==128): inactive → k_offset == v_offset == 0
  auto offsets = layout.GetKvCacheLayerOffsets(0, "attention");
  EXPECT_EQ(offsets.k_offset, offsets.v_offset);
  EXPECT_EQ(offsets.k_offset, 0u);
}

TEST_F(DeepSeekV4CompressorIndexerStateLayoutTest, OffsetLayoutMatchesManualCalculation) {
  // Verify exact offset values for a mixed-ratio configuration.
  // ratios = {0, 4, 128, 4}
  // coff=2 (block_compress_ratio==4), head_dim=index_head_dim=256
  // Per active layer: 2 * 4 * 2 * 256 * 4 = 16384 bytes
  //   Layer 0 (ratio=0):   inactive; offset=0
  //   Layer 1 (ratio=4):   active;   offset=0; next_base = 16384
  //   Layer 2 (ratio=128): inactive; offset=16384
  //   Layer 3 (ratio=4):   active;   offset=16384
  const uint32_t num_layer = 4;
  std::vector<uint32_t> ratios = {0u, 4u, 128u, 4u};
  const uint32_t index_head_dim = 256;
  const size_t block_token_num = 16;

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, index_head_dim);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(block_token_num, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, 4u, true);

  const size_t per_active_layer = 2u * 4u * 2u * index_head_dim * GetTypeSize(TYPE_FP32);

  // Layer 0 (inactive)
  {
    auto off = layout.GetKvCacheLayerOffsets(0, "attention");
    EXPECT_EQ(off.k_offset, 0u);
    EXPECT_EQ(off.v_offset, 0u);
  }
  // Layer 1 (active, first)
  {
    auto off = layout.GetKvCacheLayerOffsets(1, "attention");
    EXPECT_EQ(off.k_offset, 0u);
    EXPECT_EQ(off.v_offset, 0u);
  }
  // Layer 2 (inactive, base = per_active_layer)
  {
    auto off = layout.GetKvCacheLayerOffsets(2, "attention");
    EXPECT_EQ(off.k_offset, per_active_layer);
    EXPECT_EQ(off.v_offset, per_active_layer);
  }
  // Layer 3 (active, second)
  {
    auto off = layout.GetKvCacheLayerOffsets(3, "attention");
    EXPECT_EQ(off.k_offset, per_active_layer);
    EXPECT_EQ(off.v_offset, per_active_layer);
  }
}

TEST_F(DeepSeekV4CompressorIndexerStateLayoutTest, GetActiveKvCacheTypesReturnsAttention) {
  std::vector<uint32_t> ratios = {4u, 128u};
  auto model_cfg = MakeDsv4ModelConfig(2, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(2);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, 4u, true);
  auto types = layout.GetActiveKvCacheTypes();
  ASSERT_EQ(types.size(), 1u);
  EXPECT_EQ(types[0], "attention");
}

TEST_F(DeepSeekV4CompressorIndexerStateLayoutTest, OutOfRangeLayerIndexThrows) {
  const uint32_t num_layer = 2;
  std::vector<uint32_t> ratios(num_layer, 4u);
  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout(model_cfg, pipeline_cfg, runtime_cfg, 4u, true);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(static_cast<int32_t>(num_layer), "attention"), std::runtime_error);
  EXPECT_THROW(layout.GetKvCacheLayerOffsets(-1, "attention"), std::runtime_error);
}

TEST_F(DeepSeekV4CompressorIndexerStateLayoutTest, MtpLayersBeyondMainLayersAreInactive) {
  // MTP layers have compress_ratio == 0, so with block_compress_ratio == 4
  // those MTP layers are inactive and don't add to block size.
  const uint32_t num_layer = 2;
  const uint32_t nextn_layers = 2;
  // Pad ratios to cover MTP layers (they get ratio=0).
  std::vector<uint32_t> ratios = {4u, 4u, 0u, 0u};

  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  model_cfg.num_nextn_predict_layers = nextn_layers;

  auto pipeline_no_mtp = MakePipelineConfig(num_layer);
  auto pipeline_with_mtp = MakePipelineConfigWithMtp(num_layer, nextn_layers);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  DeepSeekV4CompressorStateLayout layout_no_mtp(model_cfg, pipeline_no_mtp, runtime_cfg, 4u, true);
  DeepSeekV4CompressorStateLayout layout_with_mtp(model_cfg, pipeline_with_mtp, runtime_cfg, 4u, true);

  // MTP layers have ratio==0, so they are inactive; block size must be the same.
  EXPECT_EQ(layout_no_mtp.GetBlockCacheSize(), layout_with_mtp.GetBlockCacheSize());
}

// ============================================================
// CacheLayoutFactory tests
// ============================================================

class CacheLayoutFactoryTest : public ::testing::Test {};

TEST_F(CacheLayoutFactoryTest, MhaModelConfigCreatesMhaLayout) {
  const uint32_t num_layer = 4;
  auto model_cfg = MakeMhaModelConfig(num_layer, 8, 128);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMhaRuntimeConfig(16);

  auto layout =
      CacheLayoutFactory::CreateModelCacheLayout(model_cfg, pipeline_cfg, runtime_cfg)->GetBlockCacheLayout("kv-cache");
  ASSERT_NE(layout, nullptr);

  // MHA layout returns only "attention".
  auto types = layout->GetActiveKvCacheTypes();
  ASSERT_EQ(types.size(), 1u);
  EXPECT_EQ(types[0], "attention");
  EXPECT_GT(layout->GetBlockCacheSize(), 0u);
}

TEST_F(CacheLayoutFactoryTest, MlaModelConfigCreatesMlaLayout) {
  const uint32_t num_layer = 4;
  auto model_cfg = MakeMlaModelConfig(num_layer, 512, 64);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  auto layout =
      CacheLayoutFactory::CreateModelCacheLayout(model_cfg, pipeline_cfg, runtime_cfg)->GetBlockCacheLayout("kv-cache");
  ASSERT_NE(layout, nullptr);

  auto types = layout->GetActiveKvCacheTypes();
  ASSERT_EQ(types.size(), 1u);
  EXPECT_EQ(types[0], "attention");

  // MLA: K and V share the same offset.
  auto offsets = layout->GetKvCacheLayerOffsets(0, "attention");
  EXPECT_EQ(offsets.k_offset, offsets.v_offset);
}

TEST_F(CacheLayoutFactoryTest, DsaModelConfigCreatesDsaLayout) {
  const uint32_t num_layer = 4;
  auto model_cfg = MakeDsaModelConfig(num_layer, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(16, TYPE_FP16);

  auto layout =
      CacheLayoutFactory::CreateModelCacheLayout(model_cfg, pipeline_cfg, runtime_cfg)->GetBlockCacheLayout("kv-cache");
  ASSERT_NE(layout, nullptr);

  auto types = layout->GetActiveKvCacheTypes();
  ASSERT_EQ(types.size(), 2u);
  EXPECT_EQ(types[0], "indexer");
  EXPECT_EQ(types[1], "attention");
}

TEST_F(CacheLayoutFactoryTest, Dsv4ModelConfigCreatesDsv4Layout) {
  const uint32_t num_layer = 4;
  std::vector<uint32_t> ratios(num_layer, 128u);
  auto model_cfg = MakeDsv4ModelConfig(num_layer, ratios, 512, 64, 256);
  auto pipeline_cfg = MakePipelineConfig(num_layer);
  auto runtime_cfg = MakeMlaRuntimeConfig(128, TYPE_FP16);

  auto layout =
      CacheLayoutFactory::CreateModelCacheLayout(model_cfg, pipeline_cfg, runtime_cfg)->GetBlockCacheLayout("kv-cache");
  ASSERT_NE(layout, nullptr);

  auto types = layout->GetActiveKvCacheTypes();
  ASSERT_EQ(types.size(), 2u);
  EXPECT_EQ(types[0], "indexer");
  EXPECT_EQ(types[1], "attention");
}

TEST_F(CacheLayoutFactoryTest, DefaultConstructorLayoutThrowsOnVirtualMethods) {
  // The default-constructed BaseBlockCacheLayout throws on all virtual methods.
  BaseBlockCacheLayout base;
  ModelConfig mc;
  PipelineConfig pc;
  RuntimeConfig rc;
  EXPECT_THROW(base.Initialize(mc, pc, rc), std::runtime_error);
  EXPECT_THROW(base.GetBlockCacheSize(), std::runtime_error);
  EXPECT_THROW(base.GetKvCacheLayerOffsets(0, "attention"), std::runtime_error);
  EXPECT_THROW(base.GetActiveKvCacheTypes(), std::runtime_error);
}

}  // namespace ksana_llm
