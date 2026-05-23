/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "cache_manager/cache_layout/cache_layout_factory.h"
#include "configure/environment.h"
#include "configure/schedule_config_parser.h"

namespace ksana_llm {

inline std::shared_ptr<BaseModelCacheLayout> GetTestModelCacheLayout(std::shared_ptr<Environment> env) {
  ModelConfig model_config;
  env->GetModelConfig(model_config);

  PipelineConfig pipeline_config;
  env->GetPipelineConfig(pipeline_config);

  RuntimeConfig runtime_config;
  env->GetRuntimeConfig(runtime_config);

  return CacheLayoutFactory::CreateModelCacheLayout(model_config, pipeline_config, runtime_config);
}

}  // namespace ksana_llm
