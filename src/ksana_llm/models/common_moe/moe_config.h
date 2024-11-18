/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/common_config.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

// The moe experts norm mode.
enum class MoeScaleNormMode { NO_NORM = 0, RE_NORM = 1 };

inline void PrepareQwen2MoeAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  PrepareCommonModelAttributes(config_json, model_config);
  model_config.moe_config.num_experts = config_json.value("num_experts", 1);
  if (model_config.moe_config.num_experts > 1) {
    model_config.is_moe = true;
    model_config.moe_config.moe_inter_size =
        config_json.value("moe_intermediate_size", model_config.moe_config.moe_inter_size);  // qwen2-moe model
    if (config_json.contains("shared_expert_intermediate_size")) {
      model_config.has_shared_experts = true;
    }
    model_config.moe_config.shared_expert_inter_size = config_json.value("shared_expert_intermediate_size", 20480);
    model_config.moe_config.experts_topk = config_json.value("num_experts_per_tok", 2);
    KLLM_LOG_INFO << fmt::format("using moe model, num_experts: {}", model_config.moe_config.num_experts);
  }
}

inline void PrepareMixtralAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  PrepareCommonModelAttributes(config_json, model_config);
  model_config.moe_config.num_experts = config_json.value("num_local_experts", 1);
  if (model_config.moe_config.num_experts > 1) {
    model_config.is_moe = true;
    model_config.has_shared_experts = false;
    model_config.moe_config.moe_inter_size = model_config.inter_size;  //  for mixtral model
    model_config.moe_config.experts_topk = config_json.value("num_experts_per_tok", 2);
    KLLM_LOG_INFO << fmt::format("using moe model, num_experts: {}", model_config.moe_config.num_experts);
  }
}

}  // namespace ksana_llm