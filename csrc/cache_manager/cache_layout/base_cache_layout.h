/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"

namespace ksana_llm {

// Per-layer K/V byte offsets within one block (absolute, already include the layer base offset).
struct KvCacheLayerOffsets {
  size_t k_offset = 0;
  size_t v_offset = 0;
};

// The base class of cache layout of one layer.
class BaseLayerCacheLayout {
 public:
  BaseLayerCacheLayout() {}
  virtual ~BaseLayerCacheLayout() {}

  // The layer index, for easily usage.
  int32_t layer_index = 0;

  // The base offset of current layer.
  size_t base_offset = 0;

  // The total cache size of layer, in byte.
  size_t cache_size = 0;
};

// The base class of cache layout of all layers in one block.
class BaseBlockCacheLayout {
 public:
  BaseBlockCacheLayout() {}
  virtual ~BaseBlockCacheLayout() {}

  // calc the block cache layout, invoked only once.
  virtual void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                          const RuntimeConfig& runtime_config) {
    throw std::runtime_error("Not implemented");
  }

  // Calculate the size of each cache block, in bytes.
  // The pipeline_config is used for distributed mode.
  virtual size_t GetBlockCacheSize() { throw std::runtime_error("Not implemented"); }

  // Return the K/V byte offsets for the given kv_cache_type and layer_index.
  // Offsets are absolute within one block (already include the layer base offset).
  virtual KvCacheLayerOffsets GetKvCacheLayerOffsets(int32_t layer_index, const std::string& kv_cache_type) {
    throw std::runtime_error("Not implemented");
  }

  // Return the list of active kv_cache_type strings for this layout.
  // The order matches the layout creation order in InitializeKVCacheConfigs:
  // indexer (lower offset) is processed before attention (higher offset).
  virtual std::vector<std::string> GetActiveKvCacheTypes() { throw std::runtime_error("Not implemented"); }
};

class BaseModelCacheLayout {
 public:
  BaseModelCacheLayout() {}
  virtual ~BaseModelCacheLayout() {}

  // Initialize all registered block cache layouts with the given configs.
  // Derived classes override this to (re-)initialize their underlying BlockCacheLayout.
  virtual void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                          const RuntimeConfig& runtime_config) {
    throw std::runtime_error("BaseModelCacheLayout::Initialize not implemented");
  }

  std::shared_ptr<BaseBlockCacheLayout> GetBlockCacheLayout(const std::string& block_cache_type) {
    if (block_cache_layouts_.find(block_cache_type) == block_cache_layouts_.end()) {
      throw std::runtime_error(
          fmt::format("BaseModelCacheLayout: The block_cache_type {} is not found.", block_cache_type));
    }
    return block_cache_layouts_[block_cache_type];
  }

  bool IsBlockCacheTypeExist(const std::string& block_cache_type) {
    return block_cache_layouts_.find(block_cache_type) != block_cache_layouts_.end();
  }

 protected:
  // Register a block cache layout under the given type name.
  // Derived classes call this in their constructor to populate the layout map.
  void RegisterBlockCacheLayout(const std::string& block_cache_type, std::shared_ptr<BaseBlockCacheLayout> layout) {
    block_cache_layouts_[block_cache_type] = layout;
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<BaseBlockCacheLayout>> block_cache_layouts_;
};
}  // namespace ksana_llm
