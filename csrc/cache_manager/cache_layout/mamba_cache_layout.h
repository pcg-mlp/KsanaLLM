/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <algorithm>
#include <cstddef>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "cache_manager/cache_layout/mha_cache_layout.h"
#include "cache_manager/cache_layout/mla_cache_layout.h"
#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "utils/logger.h"

namespace ksana_llm {

// 单个 mamba block 布局，与 Mha/Mla 同构。linear-attn state 的真实尺寸由
// linear-attn 布局单独实现（待重构），Initialize 仅注册占位，block 尺寸 0。
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
    // linear-attn state 尺寸由 linear-attn 布局单独实现（待重构）；此处仅占位。
    block_cache_size_ = 0;
  }

 private:
  size_t block_cache_size_ = 0;
};

// 混合 LA 模型的 model 级布局：kv-cache 与 mamba-state 均无条件注册。
// kv-cache 尺寸由对应 BlockCacheLayout 从 config 推导；mamba-state 尺寸待
// linear-attn 布局单独实现（当前占位 0）。
class MambaModelCacheLayout : public BaseModelCacheLayout {
 public:
  // 无配置路径（executor）：只注册存在性，真实尺寸由 engine 侧提供。
  MambaModelCacheLayout() {
    RegisterBlockCacheLayout("kv-cache", std::make_shared<MhaBlockCacheLayout>());
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
    // --- "kv-cache"：FA。混合模型的 FA 层是标准 MHA/MLA，复用通用 BlockCacheLayout
    // （不假定 MHA，MLA 层走 Mla 布局）；linear-attn 内部逐层布局是模型侧
    // linear-attn 布局的职责（待实现）。----
    if (model_config.use_mla) {
      RegisterBlockCacheLayout("kv-cache",
                               std::make_shared<MlaBlockCacheLayout>(model_config, pipeline_config, runtime_config));
    } else {
      RegisterBlockCacheLayout("kv-cache",
                               std::make_shared<MhaBlockCacheLayout>(model_config, pipeline_config, runtime_config));
    }

    // --- "mamba-state"：LA -------------------------------------------
    RegisterBlockCacheLayout("mamba-state",
                             std::make_shared<MambaBlockCacheLayout>(model_config, pipeline_config, runtime_config));
  }
};
}  // namespace ksana_llm
