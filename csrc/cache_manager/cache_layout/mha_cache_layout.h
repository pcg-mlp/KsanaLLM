/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

class MhaLayerCacheLayout : public BaseLayerCacheLayout {
 public:
  MhaLayerCacheLayout() {}
  virtual ~MhaLayerCacheLayout() {}

  size_t k_offset = 0;
  size_t v_offset = 0;
};

// The cache layout of MHA.
class MhaBlockCacheLayout : public BaseBlockCacheLayout {
 public:
  MhaBlockCacheLayout() {}
  MhaBlockCacheLayout(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                      const RuntimeConfig& runtime_config) {
    Initialize(model_config, pipeline_config, runtime_config);
  }
  virtual ~MhaBlockCacheLayout() {}

  virtual size_t GetBlockCacheSize() override { return block_cache_size_; }

  virtual KvCacheLayerOffsets GetKvCacheLayerOffsets(int32_t layer_index, const std::string& kv_cache_type) override {
    if (layer_index < 0 || layer_index >= static_cast<int>(layer_cache_layouts_.size())) {
      throw std::runtime_error(fmt::format("MhaBlockCacheLayout: The layer_index {} is exceed layerlayout_size {}.",
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
      std::shared_ptr<MhaLayerCacheLayout> mha_layer_cache_layout = std::make_shared<MhaLayerCacheLayout>();
      mha_layer_cache_layout->base_offset = mla_layer_base_offset;
      mha_layer_cache_layout->layer_index = layer_index;

      InitLayerCacheLayout(model_config, runtime_config, mha_layer_cache_layout);
      mla_layer_base_offset += mha_layer_cache_layout->cache_size;

      layer_cache_layouts_.push_back(mha_layer_cache_layout);
    }
    block_cache_size_ = mla_layer_base_offset;
  }

 private:
  void InitLayerCacheLayout(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                            std::shared_ptr<MhaLayerCacheLayout> mha_layer_cache_layout) {
    // the layer cache layout is:
    // [key:
    //          {block_token_num, head_num, head_dim}]
    // [value:
    //         {block_token_num, head_num, head_dim}]
    const size_t key_value_head_num =
        DivRoundUp(model_config.num_key_value_heads, runtime_config.parallel_basic_config.attn_tensor_parallel_size);

    const DataType& kv_cache_dtype = runtime_config.attn_backend_config.kv_cache_dtype;
    const size_t key_heads_bytes = model_config.size_per_head * GetTypeSize(kv_cache_dtype);
    const size_t value_heads_bytes = model_config.size_per_head * GetTypeSize(kv_cache_dtype);

    const size_t block_token_bytes = key_value_head_num * (key_heads_bytes + value_heads_bytes);
    mha_layer_cache_layout->cache_size = block_token_bytes * runtime_config.attn_backend_config.block_token_num;

    mha_layer_cache_layout->k_offset = mha_layer_cache_layout->base_offset;
    mha_layer_cache_layout->v_offset = mha_layer_cache_layout->k_offset + (mha_layer_cache_layout->cache_size / 2);
  }

 private:
  size_t block_cache_size_ = 0;
  std::vector<std::shared_ptr<MhaLayerCacheLayout>> layer_cache_layouts_;
};

// The model-level cache layout for MHA, registers MhaBlockCacheLayout under "kv-cache".
class MhaModelCacheLayout : public BaseModelCacheLayout {
 public:
  MhaModelCacheLayout() { RegisterBlockCacheLayout("kv-cache", std::make_shared<MhaBlockCacheLayout>()); }
  MhaModelCacheLayout(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                      const RuntimeConfig& runtime_config) {
    RegisterBlockCacheLayout("kv-cache",
                             std::make_shared<MhaBlockCacheLayout>(model_config, pipeline_config, runtime_config));
  }
  virtual ~MhaModelCacheLayout() {}

  void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                  const RuntimeConfig& runtime_config) override {
    RegisterBlockCacheLayout("kv-cache",
                             std::make_shared<MhaBlockCacheLayout>(model_config, pipeline_config, runtime_config));
  }
};

}  // namespace ksana_llm
