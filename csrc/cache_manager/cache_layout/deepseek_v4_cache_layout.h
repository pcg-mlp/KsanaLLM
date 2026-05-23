/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "utils/logger.h"

namespace ksana_llm {

// Compute the uncompressed indexer key/value block sizes for a single layer.
// Caller guards compress_ratio == 4 before invoking this function.
// Indexer layout per token:
//   key:   index_head_dim bytes  (fp8_e4m3 quant)
//   value: index_head_dim / 128 * sizeof(fp32)  (fp32 scale)
// Output parameters (without compress_ratio applied):
//   key_bytes   = key_head_bytes * block_token_num
//   value_bytes = value_head_bytes * block_token_num
inline void ComputeIndexerCacheSize(const ModelConfig& model_config, size_t block_token_num, size_t& key_bytes,
                                    size_t& value_bytes) {
  const DataType indexer_kv_cache_dtype = TYPE_FP8_E4M3;
  constexpr size_t QUANT_BLOCK_SIZE = 128;
  const size_t key_head_bytes = model_config.dsa_config.index_head_dim * GetTypeSize(indexer_kv_cache_dtype);
  const size_t value_head_bytes = model_config.dsa_config.index_head_dim / QUANT_BLOCK_SIZE * GetTypeSize(TYPE_FP32);
  key_bytes = key_head_bytes * block_token_num;
  value_bytes = value_head_bytes * block_token_num;
}

// Compute the total MLA attention cache size (in bytes) for a single layer token block.
// TYPE_FP8_DS_MLA path:
//   block_token_num x (kv_lora_rank<fp8 quant> + kv_lora_rank / 128<fp32 scale> + qk_rope_head_dim<weight dtype>)
// Default path:
//   block_token_num x (kv_lora_rank + qk_rope_head_dim) * sizeof(kv_cache_dtype)
// The caller is responsible for dividing by compress_ratio to obtain the compressed size.
inline size_t ComputeKeyValueHeadBytes(const ModelConfig& model_config, const DataType& mla_kv_cache_dtype,
                                       size_t block_token_num) {
  if (mla_kv_cache_dtype == TYPE_FP8_DS_MLA) {
    constexpr size_t QUANT_BLOCK_SIZE = 64;
    const uint32_t qk_nope_head_dim = model_config.size_per_head - model_config.dsv4_attn_config.qk_rope_head_dim;
    return (qk_nope_head_dim * GetTypeSize(TYPE_FP8_E4M3) +
            (qk_nope_head_dim / QUANT_BLOCK_SIZE + 1 /*for padding*/) * GetTypeSize(TYPE_UINT8) /*ue8m0*/
            + model_config.dsv4_attn_config.qk_rope_head_dim * GetTypeSize(model_config.weight_data_type)) *
           block_token_num;
  }
  return model_config.size_per_head * GetTypeSize(mla_kv_cache_dtype) * block_token_num;
}

// The per-layer cache layout of a DeepSeek-V4 layer.
// Each layer may have three structural variants depending on compress_ratio:
//   compress_ratio == 0   : no indexer, no attention cache (both sizes are 0)
//   compress_ratio == 4   : indexer cache present; attention cache size = DSA size / 4
//   compress_ratio == 128 : no indexer cache; attention cache size = DSA size / 128
//
// 主 KV layout (kv-cache) 每层仅持有 [indexer | attention] 两段。
// swa / tail_attn / tail_indexer 由独立的 DeepSeekV4SwaWithTailCacheLayout
// (block_cache_type = "SWA-cache") 管理。
class DeepSeekV4LayerCacheLayout : public BaseLayerCacheLayout {
 public:
  DeepSeekV4LayerCacheLayout() {}
  virtual ~DeepSeekV4LayerCacheLayout() {}

  size_t indexer_k_offset = 0;
  size_t indexer_v_offset = 0;
  size_t attention_k_offset = 0;
  size_t attention_v_offset = 0;
};

// SWA + c4 overlap tail layout, 一个 SWA block 的单层布局:
//   [swa | tail_attn | tail_indexer]
// 每段语义 (与主 KV 拆开后单独管理):
//   swa          : full-size MLA cache (block_token_num x head_dim)，所有层均匀分配。
//   tail_attn    : c4 overlap tail (main 路径)，固定 4 个 token (kC4Ratio) 的 FP32 kv_score。
//   tail_indexer : c4 overlap tail (indexer 路径)，固定 4 个 token 的 FP32 kv_score。
//
// 所有层均匀分配，c128 / MTP 层 tail 区占位不读写，方便单 base + 固定 stride 寻址。
class DeepSeekV4SwaLayerCacheLayout : public BaseLayerCacheLayout {
 public:
  DeepSeekV4SwaLayerCacheLayout() {}
  virtual ~DeepSeekV4SwaLayerCacheLayout() {}

  size_t swa_k_offset = 0;
  size_t swa_v_offset = 0;
  size_t tail_attn_offset = 0;
  size_t tail_attn_size = 0;
  size_t tail_indexer_offset = 0;
  size_t tail_indexer_size = 0;
};

// The block-level cache layout for DeepSeek-V4 series (v4-Flash and v4-Pro).
// Replaces the previous hard-coded placeholder with a per-layer calculation
// driven by model_config.dsv4_attn_config.compress_ratios[layer_idx].
class DeepSeekV4CacheLayout : public BaseBlockCacheLayout {
 public:
  DeepSeekV4CacheLayout() {}
  DeepSeekV4CacheLayout(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                        const RuntimeConfig& runtime_config) {
    Initialize(model_config, pipeline_config, runtime_config);
  }
  virtual ~DeepSeekV4CacheLayout() {}

  virtual size_t GetBlockCacheSize() override { return block_cache_size_; }

  virtual KvCacheLayerOffsets GetKvCacheLayerOffsets(int32_t layer_index, const std::string& kv_cache_type) override {
    if (layer_index < 0 || layer_index >= static_cast<int>(layer_cache_layouts_.size())) {
      throw std::runtime_error(fmt::format("DeepSeekV4CacheLayout: The layer_index {} exceeds layerlayout_size {}.",
                                           layer_index, layer_cache_layouts_.size()));
    }
    const auto& layer = layer_cache_layouts_[layer_index];
    if (kv_cache_type == "attention") {
      return {layer->attention_k_offset, layer->attention_v_offset};
    }
    // 主 KV layout 不持有 swa / tail_attn / tail_indexer；
    // 这些字段由 "SWA-cache" 注册项 (DeepSeekV4SwaWithTailCacheLayout) 管理。
    if (kv_cache_type == "swa" || kv_cache_type == "tail_attn" || kv_cache_type == "tail_indexer") {
      throw std::runtime_error(
          fmt::format("DeepSeekV4CacheLayout: kv_cache_type '{}' has been migrated to the SWA-cache layout. "
                      "Use GetBlockCacheLayout(\"SWA-cache\") instead.",
                      kv_cache_type));
    }
    // kv_cache_type == "indexer"
    return {layer->indexer_k_offset, layer->indexer_v_offset};
  }

  // indexer (lower offset) is processed before attention (higher offset),
  // matching the layout creation order in InitializeKVCacheConfigs.
  virtual std::vector<std::string> GetActiveKvCacheTypes() override { return {"indexer", "attention"}; }

  // Calculate the block cache layout; invoked only once.
  void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                  const RuntimeConfig& runtime_config) override {
    // MTP part.
    const bool predict_nextn =
        static_cast<int>(pipeline_config.lower_nextn_layer_idx) >= static_cast<int>(model_config.num_layer);
    const size_t node_nextn_layer_num =
        predict_nextn ? pipeline_config.upper_nextn_layer_idx - pipeline_config.lower_nextn_layer_idx + 1 : 0;

    // All layers including MTP.
    const size_t node_layer_num =
        pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + 1 + node_nextn_layer_num;

    size_t layer_base_offset = 0;
    for (size_t layer_index = pipeline_config.lower_layer_idx;
         layer_index < (pipeline_config.lower_layer_idx + node_layer_num); ++layer_index) {
      std::shared_ptr<DeepSeekV4LayerCacheLayout> layer_cache_layout = std::make_shared<DeepSeekV4LayerCacheLayout>();
      layer_cache_layout->base_offset = layer_base_offset;
      layer_cache_layout->layer_index = static_cast<int32_t>(layer_index);

      InitLayerCacheLayout(model_config, runtime_config, layer_cache_layout);
      layer_base_offset += layer_cache_layout->cache_size;

      layer_cache_layouts_.push_back(layer_cache_layout);
    }
    block_cache_size_ = layer_base_offset;
  }

 private:
  // Compute the indexer and attention offsets/sizes for a single layer.
  // The compress_ratio value determines which of the three structural variants applies:
  //   0   -> both indexer and attention cache sizes are 0
  //   4   -> indexer cache present; attention cache size divided by compress_ratio
  //   128 -> indexer cache size is 0; attention cache size divided by compress_ratio
  void InitLayerCacheLayout(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                            std::shared_ptr<DeepSeekV4LayerCacheLayout> layer_cache_layout) {
    const size_t layer_index = static_cast<size_t>(layer_cache_layout->layer_index);
    const auto& compress_ratios = model_config.dsv4_attn_config.compress_ratios;

    const uint32_t compress_ratio = compress_ratios[layer_index];

    // 当 compress_ratio == 0, layer cache_size 按公式计算 (indexer=0 + attn=0 = 0)；
    // SWA/tail 由 DeepSeekV4SwaWithTailCacheLayout 统一管理, 不在主 KV layout 中。
    const size_t block_token_num = runtime_config.attn_backend_config.block_token_num;
    const size_t compressed_block_token_num = compress_ratio == 0 ? 0 : block_token_num / compress_ratio;

    // ---- Indexer part ----
    // The indexer is present only when compress_ratio == 4.
    size_t indexer_cache_size = 0;
    if (compress_ratio == 4) {
      size_t indexer_key_bytes = 0;
      size_t indexer_value_bytes = 0;
      ComputeIndexerCacheSize(model_config, compressed_block_token_num, indexer_key_bytes, indexer_value_bytes);
      layer_cache_layout->indexer_k_offset = layer_cache_layout->base_offset;
      layer_cache_layout->indexer_v_offset = layer_cache_layout->indexer_k_offset + indexer_key_bytes;
      indexer_cache_size = (indexer_key_bytes + indexer_value_bytes);
    } else {
      // compress_ratio == 128: no indexer cache; offsets point to base.
      layer_cache_layout->indexer_k_offset = layer_cache_layout->base_offset;
      layer_cache_layout->indexer_v_offset = layer_cache_layout->base_offset;
    }
    // ---- Attention (MLA) part ----
    // Layout (same as DsaBlockCacheLayout) but the total size is divided by compress_ratio.
    // TYPE_FP8_DS_MLA path:
    //   [key/value: block_token_num x (kv_lora_rank<fp8 quant> +
    //                                  kv_lora_rank / 128<fp32 scale> +
    //                                  qk_rope_head_dim)]
    // Default path:
    //   [key/value: block_token_num x (kv_lora_rank + qk_rope_head_dim)]
    const DataType& mla_kv_cache_dtype = runtime_config.attn_backend_config.kv_cache_dtype;
    const size_t key_value_head_bytes =
        ComputeKeyValueHeadBytes(model_config, mla_kv_cache_dtype, compressed_block_token_num);

    // Divide by compress_ratio to obtain the compressed attention cache size.
    const size_t attention_cache_size = key_value_head_bytes;

    // The key and value share the same offset (MLA property).
    layer_cache_layout->attention_k_offset = layer_cache_layout->indexer_k_offset + indexer_cache_size;
    layer_cache_layout->attention_v_offset = layer_cache_layout->attention_k_offset;

    // 主 KV layout 仅含 indexer + attention; swa/tail 由 DeepSeekV4SwaWithTailCacheLayout 管理。
    layer_cache_layout->cache_size = indexer_cache_size + attention_cache_size;

    KLLM_LOG_DEBUG << fmt::format(
        "DeepSeekV4 layer {} kv-cache layout: indexer=[{},{}) attn=[{},{}) cache_size={}", layer_index,
        layer_cache_layout->indexer_k_offset, layer_cache_layout->indexer_k_offset + indexer_cache_size,
        layer_cache_layout->attention_k_offset, layer_cache_layout->attention_k_offset + attention_cache_size,
        layer_cache_layout->cache_size);
  }

 private:
  size_t block_cache_size_ = 0;
  std::vector<std::shared_ptr<DeepSeekV4LayerCacheLayout>> layer_cache_layouts_;
};

// The block-level cache layout for SWA + c4 overlap tail (DeepSeek-V4).
// 单 SWA block 的总布局: 所有层连续排列，每层内顺序为
//   [swa | tail_attn | tail_indexer]
// 其中 swa 持有 full-size MLA cache，tail_attn / tail_indexer 持有 4 token kv_score。
//
// 注: c128 / MTP 层 swa+tail 仍按统一公式分配 (compressor 的 c128 路径不读 tail，
// 由 caller 通过 compress_ratio 守卫；统一公式保证 per-layer offset 跨层一致便于
// 单 base + 固定 stride 寻址)。
class DeepSeekV4SwaWithTailCacheLayout : public BaseBlockCacheLayout {
 public:
  DeepSeekV4SwaWithTailCacheLayout() {}
  DeepSeekV4SwaWithTailCacheLayout(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                                   const RuntimeConfig& runtime_config) {
    Initialize(model_config, pipeline_config, runtime_config);
  }
  virtual ~DeepSeekV4SwaWithTailCacheLayout() {}

  virtual size_t GetBlockCacheSize() override { return block_cache_size_; }

  virtual KvCacheLayerOffsets GetKvCacheLayerOffsets(int32_t layer_index, const std::string& kv_cache_type) override {
    if (layer_index < 0 || layer_index >= static_cast<int>(layer_cache_layouts_.size())) {
      throw std::runtime_error(
          fmt::format("DeepSeekV4SwaWithTailCacheLayout: The layer_index {} exceeds layerlayout_size {}.", layer_index,
                      layer_cache_layouts_.size()));
    }
    const auto& layer = layer_cache_layouts_[layer_index];
    if (kv_cache_type == "swa") {
      return {layer->swa_k_offset, layer->swa_v_offset};
    }
    if (kv_cache_type == "tail_attn") {
      return {layer->tail_attn_offset, layer->tail_attn_offset};
    }
    if (kv_cache_type == "tail_indexer") {
      return {layer->tail_indexer_offset, layer->tail_indexer_offset};
    }
    throw std::runtime_error(fmt::format(
        "DeepSeekV4SwaWithTailCacheLayout: kv_cache_type '{}' not supported (expected swa / tail_attn / tail_indexer).",
        kv_cache_type));
  }

  // SWA layout 不参与 model_input.cpp 的 PrepareKVCacheBlocks 循环
  // (Dsv4MqaMeta 直接通过 forward_req.swa_block_base_addr + 静态 layout offset 路径访问)，
  // 所以 GetActiveKvCacheTypes 返回空表，避免 model_input 把 swa/tail 当成主 KV 插入 kv_list。
  virtual std::vector<std::string> GetActiveKvCacheTypes() override { return {}; }

  void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                  const RuntimeConfig& runtime_config) override {
    const bool predict_nextn =
        static_cast<int>(pipeline_config.lower_nextn_layer_idx) >= static_cast<int>(model_config.num_layer);
    const size_t node_nextn_layer_num =
        predict_nextn ? pipeline_config.upper_nextn_layer_idx - pipeline_config.lower_nextn_layer_idx + 1 : 0;
    const size_t node_layer_num =
        pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + 1 + node_nextn_layer_num;

    const size_t block_token_num = runtime_config.attn_backend_config.block_token_num;
    const DataType& mla_kv_cache_dtype = runtime_config.attn_backend_config.kv_cache_dtype;
    const size_t swa_cache_size = ComputeKeyValueHeadBytes(model_config, mla_kv_cache_dtype, block_token_num);

    // c4 overlap tail: 容量固定 4 个 token (c4 overlap 窗口)。Per-token 字节数:
    //   FP32 kv_score [2 (kv+score) * coff (=2 for c4 overlap) * head_dim] = 16 * head_dim
    constexpr size_t kTailTokens = 4;   // tokens per c4 group
    constexpr size_t kCoffMax = 2;      // c4 overlap window
    constexpr size_t kKvScorePair = 2;  // kv FP32 + score FP32
    const size_t per_token_attn_bytes = kKvScorePair * kCoffMax * model_config.size_per_head * GetTypeSize(TYPE_FP32);
    const size_t per_token_idx_bytes =
        kKvScorePair * kCoffMax * model_config.dsa_config.index_head_dim * GetTypeSize(TYPE_FP32);
    const size_t tail_attn_size = kTailTokens * per_token_attn_bytes;
    const size_t tail_indexer_size = kTailTokens * per_token_idx_bytes;

    size_t layer_base_offset = 0;
    for (size_t layer_index = pipeline_config.lower_layer_idx;
         layer_index < (pipeline_config.lower_layer_idx + node_layer_num); ++layer_index) {
      auto layer_cache_layout = std::make_shared<DeepSeekV4SwaLayerCacheLayout>();
      layer_cache_layout->base_offset = layer_base_offset;
      layer_cache_layout->layer_index = static_cast<int32_t>(layer_index);

      layer_cache_layout->swa_k_offset = layer_base_offset;
      layer_cache_layout->swa_v_offset = layer_base_offset;  // MLA: K/V share offset
      layer_cache_layout->tail_attn_offset = layer_base_offset + swa_cache_size;
      layer_cache_layout->tail_attn_size = tail_attn_size;
      layer_cache_layout->tail_indexer_offset = layer_cache_layout->tail_attn_offset + tail_attn_size;
      layer_cache_layout->tail_indexer_size = tail_indexer_size;
      layer_cache_layout->cache_size = swa_cache_size + tail_attn_size + tail_indexer_size;
      layer_base_offset += layer_cache_layout->cache_size;

      KLLM_LOG_DEBUG << fmt::format(
          "DeepSeekV4 layer {} SWA-cache layout: swa=[{},{}) tail_attn=[{},{}) tail_indexer=[{},{}) cache_size={}",
          layer_index, layer_cache_layout->swa_k_offset, layer_cache_layout->swa_k_offset + swa_cache_size,
          layer_cache_layout->tail_attn_offset, layer_cache_layout->tail_attn_offset + tail_attn_size,
          layer_cache_layout->tail_indexer_offset, layer_cache_layout->tail_indexer_offset + tail_indexer_size,
          layer_cache_layout->cache_size);
      layer_cache_layouts_.push_back(layer_cache_layout);
    }
    block_cache_size_ = layer_base_offset;
  }

 private:
  size_t block_cache_size_ = 0;
  std::vector<std::shared_ptr<DeepSeekV4SwaLayerCacheLayout>> layer_cache_layouts_;
};

// The block-level cache layout for a single compressor state in DeepSeek-V4.
// Each layer's cache size is determined by ComputeKeyValueHeadBytes, filtered by block_compress_ratio:
//   block_compress_ratio == 0 : every layer is allocated ComputeKeyValueHeadBytes bytes (no filtering)
//   block_compress_ratio != 0 : only layers whose layer_compress_ratio == block_compress_ratio are allocated
//                               ComputeKeyValueHeadBytes bytes; all other layers get 0
class DeepSeekV4CompressorStateLayout : public BaseBlockCacheLayout {
 public:
  DeepSeekV4CompressorStateLayout() {}
  DeepSeekV4CompressorStateLayout(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                                  const RuntimeConfig& runtime_config, uint32_t block_compress_ratio,
                                  bool is_in_indexer) {
    Initialize(model_config, pipeline_config, runtime_config, block_compress_ratio, is_in_indexer);
  }
  virtual ~DeepSeekV4CompressorStateLayout() {}

  virtual size_t GetBlockCacheSize() override { return block_cache_size_; }

  virtual KvCacheLayerOffsets GetKvCacheLayerOffsets(int32_t layer_index, const std::string& kv_cache_type) override {
    if (layer_index < 0 || layer_index >= static_cast<int>(layer_offsets_.size())) {
      throw std::runtime_error(
          fmt::format("DeepSeekV4CompressorStateLayout: The layer_index {} exceeds layerlayout_size {}.", layer_index,
                      layer_offsets_.size()));
    }
    // key and value share the same offset (MLA property).
    return {layer_offsets_[layer_index], layer_offsets_[layer_index]};
  }

  virtual std::vector<std::string> GetActiveKvCacheTypes() override { return {"attention"}; }

  // Calculate the block cache layout for the given block_compress_ratio; invoked only once.
  void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                  const RuntimeConfig& runtime_config, uint32_t block_compress_ratio, bool is_in_indexer) {
    // MTP part.
    const bool predict_nextn =
        static_cast<int>(pipeline_config.lower_nextn_layer_idx) >= static_cast<int>(model_config.num_layer);
    const size_t node_nextn_layer_num =
        predict_nextn ? pipeline_config.upper_nextn_layer_idx - pipeline_config.lower_nextn_layer_idx + 1 : 0;

    // All layers including MTP.
    const size_t node_layer_num =
        pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + 1 + node_nextn_layer_num;

    const int coff = 1 + (block_compress_ratio == 4u);
    const uint32_t head_dim = (is_in_indexer ? model_config.dsa_config.index_head_dim : model_config.size_per_head);

    const auto& compress_ratios = model_config.dsv4_attn_config.compress_ratios;

    size_t layer_base_offset = 0;
    for (size_t layer_index = pipeline_config.lower_layer_idx;
         layer_index < (pipeline_config.lower_layer_idx + node_layer_num); ++layer_index) {
      layer_offsets_.push_back(layer_base_offset);

      // Determine whether this layer is active for the given block_compress_ratio.
      const uint32_t layer_compress_ratio = compress_ratios[layer_index];
      const bool layer_active = (layer_compress_ratio == block_compress_ratio);

      if (layer_active) {
        // Multiply by 2 for kv [head_dim] FP32 and score [head_dim] FP32
        layer_base_offset += layer_compress_ratio * 2 * coff * head_dim * GetTypeSize(TYPE_FP32);
      }
    }
    block_cache_size_ = layer_base_offset;
  }

  // Overload required by BaseBlockCacheLayout; delegates to the extended signature.
  void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                  const RuntimeConfig& runtime_config) override {
    Initialize(model_config, pipeline_config, runtime_config, /*block_compress_ratio=*/0, /*is_in_indexer*/ false);
  }

 private:
  size_t block_cache_size_ = 0;
  std::vector<size_t> layer_offsets_;
};

// The model-level cache layout for DeepSeek-V4.
// Registers the following block cache layouts:
//   "kv-cache"     : DeepSeekV4CacheLayout            (per-layer indexer+attention only after SWA pool split)
//   "SWA-cache"    : DeepSeekV4SwaWithTailCacheLayout (per-layer swa + tail_attn + tail_indexer)
//   "C4-state"     : DeepSeekV4CompressorStateLayout  (block_compress_ratio == 4)
//   "C128-state"   : DeepSeekV4CompressorStateLayout  (block_compress_ratio == 128)
//   "indexer-state": DeepSeekV4CompressorIndexerStateLayout  (compress_ratio == 4 layers only)
class DeepSeekV4ModelCacheLayout : public BaseModelCacheLayout {
 public:
  DeepSeekV4ModelCacheLayout() {}
  DeepSeekV4ModelCacheLayout(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                             const RuntimeConfig& runtime_config) {
    RegisterLayouts(model_config, pipeline_config, runtime_config);
  }
  virtual ~DeepSeekV4ModelCacheLayout() {}

  void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                  const RuntimeConfig& runtime_config) override {
    RegisterLayouts(model_config, pipeline_config, runtime_config);
  }

 private:
  void RegisterLayouts(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                       const RuntimeConfig& runtime_config) {
    RegisterBlockCacheLayout("kv-cache",
                             std::make_shared<DeepSeekV4CacheLayout>(model_config, pipeline_config, runtime_config));
    // SWA-cache 持有独立的 SWA + tail pool (与主 KV pool 解耦);
    // 仅在 model 启用 sliding window (sliding_window > 0) 时才需要使用,
    // 但 layout 始终注册, 下游通过 IsBlockCacheTypeExist("SWA-cache") 判断是否启用。
    RegisterBlockCacheLayout(
        "SWA-cache", std::make_shared<DeepSeekV4SwaWithTailCacheLayout>(model_config, pipeline_config, runtime_config));
    RegisterBlockCacheLayout("C4-state", std::make_shared<DeepSeekV4CompressorStateLayout>(
                                             model_config, pipeline_config, runtime_config, 4u, false));
    RegisterBlockCacheLayout("C128-state", std::make_shared<DeepSeekV4CompressorStateLayout>(
                                               model_config, pipeline_config, runtime_config, 128u, false));
    RegisterBlockCacheLayout("indexer-state", std::make_shared<DeepSeekV4CompressorStateLayout>(
                                                  model_config, pipeline_config, runtime_config, 4u, true));
  }
};

}  // namespace ksana_llm
