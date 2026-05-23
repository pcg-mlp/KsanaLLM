/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "cache_manager/cache_layout/deepseek_v4_cache_layout.h"
#include "cache_manager/cache_layout/dsa_cache_layout.h"
#include "cache_manager/cache_layout/mha_cache_layout.h"
#include "cache_manager/cache_layout/mla_cache_layout.h"

namespace ksana_llm {

// The factory class used to create model-level cache layout.
class CacheLayoutFactory {
 public:
  static std::shared_ptr<BaseModelCacheLayout> CreateModelCacheLayout(const ModelConfig& model_config) {
    if (model_config.use_dsv4_attn) {
      return std::make_shared<DeepSeekV4ModelCacheLayout>();
    }
    if (model_config.use_dsa) {
      return std::make_shared<DsaModelCacheLayout>();
    }
    if (model_config.use_mla) {
      return std::make_shared<MlaModelCacheLayout>();
    }
    return std::make_shared<MhaModelCacheLayout>();
  }

  static std::shared_ptr<BaseModelCacheLayout> CreateModelCacheLayout(const ModelConfig& model_config,
                                                                      const PipelineConfig& pipeline_config,
                                                                      const RuntimeConfig& runtime_config) {
    if (model_config.use_dsv4_attn) {
      return std::make_shared<DeepSeekV4ModelCacheLayout>(model_config, pipeline_config, runtime_config);
    }
    if (model_config.use_dsa) {
      return std::make_shared<DsaModelCacheLayout>(model_config, pipeline_config, runtime_config);
    }
    if (model_config.use_mla) {
      return std::make_shared<MlaModelCacheLayout>(model_config, pipeline_config, runtime_config);
    }
    return std::make_shared<MhaModelCacheLayout>(model_config, pipeline_config, runtime_config);
  }
};

}  // namespace ksana_llm
