/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "models/base/base_model_config_parser.h"
#include "models/qwen2_moe/new_qwen2_moe_config.h"

namespace ksana_llm {

// Model config parser for Qwen2Moe loader.
class NewQwen2MoeConfigParser : public BaseModelConfigParser {
 public:
  // Parse config from config.json
  virtual Status ParseModelConfig(const nlohmann::json &config_json,
                                  const ParallelismBasicConfig &parallel_basic_config, const std::string &model_dir,
                                  std::shared_ptr<BaseModelConfig> &model_config) override;

 private:
  // Parse MoE config
  Status ParseMoeConfig(const nlohmann::json &config_json, std::shared_ptr<NewQwen2MoeConfig> new_qwen2_moe_config);
};

}  // namespace ksana_llm