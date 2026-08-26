/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <algorithm>
#include <cstddef>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "cache_manager/cache_layout/kda_cache_layout.h"
#include "cache_manager/cache_layout/mha_cache_layout.h"
#include "cache_manager/cache_layout/mla_cache_layout.h"
#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "utils/logger.h"

namespace ksana_llm {

class MambaBlockCacheLayout : public BaseBlockCacheLayout {
 public:
  explicit MambaBlockCacheLayout(size_t block_cache_size) : block_cache_size_(block_cache_size) {}
  MambaBlockCacheLayout(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                        const RuntimeConfig& runtime_config) {
    Initialize(model_config, pipeline_config, runtime_config);
  }
  virtual ~MambaBlockCacheLayout() {}

  size_t GetBlockCacheSize() override { return block_cache_size_; }

  void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                  const RuntimeConfig& runtime_config) override {
    block_cache_size_ = 0;
  }

 private:
  size_t block_cache_size_ = 0;
};

// 混合 LA 模型的 model 级布局：kv-cache 与 mamba-state 均无条件注册。
class MambaModelCacheLayout : public BaseModelCacheLayout {
 public:
  explicit MambaModelCacheLayout(const ModelConfig& model_config) {
    if (model_config.use_mla) {
      RegisterBlockCacheLayout("kv-cache", std::make_shared<MlaBlockCacheLayout>());
    } else {
      RegisterBlockCacheLayout("kv-cache", std::make_shared<MhaBlockCacheLayout>());
    }
    RegisterBlockCacheLayout("mamba-state", std::make_shared<MambaBlockCacheLayout>(0));
  }
  MambaModelCacheLayout(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                        const RuntimeConfig& runtime_config) {
    RegisterLayouts(model_config, pipeline_config, runtime_config);
  }
  virtual ~MambaModelCacheLayout() {}

  void Initialize(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                  const RuntimeConfig& runtime_config) override {
    RegisterLayouts(model_config, pipeline_config, runtime_config);
  }

 private:
  void RegisterLayouts(const ModelConfig& model_config, const PipelineConfig& pipeline_config,
                       const RuntimeConfig& runtime_config) {
    // --- "kv-cache"：FA。混合模型的 FA 层是标准 MHA/MLA，复用通用 BlockCacheLayout ---
    if (model_config.use_mla) {
      RegisterBlockCacheLayout("kv-cache",
                               std::make_shared<MlaBlockCacheLayout>(model_config, pipeline_config, runtime_config));
    } else {
      RegisterBlockCacheLayout("kv-cache",
                               std::make_shared<MhaBlockCacheLayout>(model_config, pipeline_config, runtime_config));
    }

    // --- "mamba-state"：(conv + recurrent) ----------------------
    if (!model_config.kimi_linear_config.kda_layers.empty()) {
      RegisterBlockCacheLayout("mamba-state",
                               std::make_shared<KDAStateLayout>(model_config, pipeline_config, runtime_config));
    } else {
      RegisterBlockCacheLayout("mamba-state", std::make_shared<MambaBlockCacheLayout>(0));
    }
  }
};

}  // namespace ksana_llm
