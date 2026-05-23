/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"

namespace ksana_llm {

// The cache layout of deepseek mla layer.
class MlaLayerCacheLayout : public BaseLayerCacheLayout {
 public:
  MlaLayerCacheLayout() {}
  virtual ~MlaLayerCacheLayout() {}

  size_t k_offset = 0;
  size_t v_offset = 0;
};

// The cache layout of DeepSeek MLA.
class MlaBlockCacheLayout : public BaseBlockCacheLayout {
 public:
  MlaBlockCacheLayout() {}
  MlaBlockCacheLayout(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                      const RuntimeConfig& runtime_config) {
    Initialize(model_config, pipeline_config, runtime_config);
  }
  virtual ~MlaBlockCacheLayout() {}

  virtual size_t GetBlockCacheSize() override { return block_cache_size_; }

  virtual KvCacheLayerOffsets GetKvCacheLayerOffsets(int32_t layer_index, const std::string& kv_cache_type) override {
    if (layer_index < 0 || layer_index >= static_cast<int>(layer_cache_layouts_.size())) {
      throw std::runtime_error(fmt::format("MlaBlockCacheLayout: The layer_index {} is exceed layerlayout_size {}.",
                                           layer_index, layer_cache_layouts_.size()));
    }
    const auto& layer = layer_cache_layouts_[layer_index];
    return {layer->k_offset, layer->v_offset};
  }

  virtual std::vector<std::string> GetActiveKvCacheTypes() override { return {"attention"}; }

  // calc the block cache layout, invoked only once.
  void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                  const RuntimeConfig& runtime_config) override {
    // MTP part.
    const bool predict_nextn =
        static_cast<int>(pipeline_config.lower_nextn_layer_idx) >= static_cast<int>(model_config.num_layer);
    const size_t node_nextn_layer_num =
        predict_nextn ? pipeline_config.upper_nextn_layer_idx - pipeline_config.lower_nextn_layer_idx + 1 : 0;

    // all layers including mtp.
    const size_t node_layer_num =
        pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + 1 + node_nextn_layer_num;

    size_t mla_layer_base_offset = 0;
    for (size_t layer_index = 0; layer_index < node_layer_num; ++layer_index) {
      std::shared_ptr<MlaLayerCacheLayout> mla_layer_cache_layout = std::make_shared<MlaLayerCacheLayout>();
      mla_layer_cache_layout->base_offset = mla_layer_base_offset;
      mla_layer_cache_layout->layer_index = layer_index;

      InitLayerCacheLayout(model_config, runtime_config, mla_layer_cache_layout);
      mla_layer_base_offset += mla_layer_cache_layout->cache_size;

      layer_cache_layouts_.push_back(mla_layer_cache_layout);
    }
    block_cache_size_ = mla_layer_base_offset;
  }

 private:
  void InitLayerCacheLayout(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                            std::shared_ptr<MlaLayerCacheLayout> mla_layer_cache_layout) {
    size_t key_value_head_bytes;
    const DataType& kv_cache_dtype = runtime_config.attn_backend_config.kv_cache_dtype;
    if (kv_cache_dtype == TYPE_FP8_DS_MLA) {
      // the layer cache layout is:
      // [key/value:
      //             block_token_num x (kv_lora_rank<fp8 quant> +
      //                                kv_lora_rank / 128<fp32 scale> +
      //                                qk_rope_head_dim)]
      constexpr size_t QUANT_BLOCK_SIZE = 128;
      key_value_head_bytes = model_config.mla_config.kv_lora_rank * GetTypeSize(TYPE_FP8_E4M3) +
                             model_config.mla_config.kv_lora_rank / QUANT_BLOCK_SIZE * GetTypeSize(TYPE_FP32) +
                             model_config.mla_config.qk_rope_head_dim * GetTypeSize(model_config.weight_data_type);
    } else {
      // The memory layout is:
      // [key/value:
      //             block_token_num x (kv_lora_rank +
      //                                qk_rope_head_dim)]
      key_value_head_bytes = (model_config.mla_config.kv_lora_rank + model_config.mla_config.qk_rope_head_dim) *
                             GetTypeSize(kv_cache_dtype);
    }

    // mla have only one head.
    mla_layer_cache_layout->cache_size = key_value_head_bytes * runtime_config.attn_backend_config.block_token_num;

    // The key and value used same offset.
    mla_layer_cache_layout->k_offset = mla_layer_cache_layout->base_offset;
    mla_layer_cache_layout->v_offset = mla_layer_cache_layout->k_offset;
  }

 private:
  size_t block_cache_size_ = 0;
  std::vector<std::shared_ptr<MlaLayerCacheLayout>> layer_cache_layouts_;
};

// The model-level cache layout for MLA, registers MlaBlockCacheLayout under "kv-cache".
class MlaModelCacheLayout : public BaseModelCacheLayout {
 public:
  MlaModelCacheLayout() { RegisterBlockCacheLayout("kv-cache", std::make_shared<MlaBlockCacheLayout>()); }
  MlaModelCacheLayout(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                      const RuntimeConfig& runtime_config) {
    RegisterBlockCacheLayout("kv-cache",
                             std::make_shared<MlaBlockCacheLayout>(model_config, pipeline_config, runtime_config));
  }
  virtual ~MlaModelCacheLayout() {}

  void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                  const RuntimeConfig& runtime_config) override {
    RegisterBlockCacheLayout("kv-cache",
                             std::make_shared<MlaBlockCacheLayout>(model_config, pipeline_config, runtime_config));
  }
};

}  // namespace ksana_llm
