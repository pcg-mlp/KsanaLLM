/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/base_model_config_parser.h"
#include "models/kimi_k3/kimi_k3_config.h"

namespace ksana_llm {

// Model config parser for kimi_k3
class KimiK3ConfigParser : public BaseModelConfigParser {
 public:
  Status ParseModelConfig(const nlohmann::json &config_json, const ParallelismBasicConfig &parallel_basic_config,
                          const std::string &model_dir, std::shared_ptr<BaseModelConfig> &model_config) override;
};

}  // namespace ksana_llm
