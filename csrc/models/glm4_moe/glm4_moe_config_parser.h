/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "models/base/base_model_config_parser.h"
#include "models/glm4_moe/glm4_moe_config.h"

namespace ksana_llm {

// Model config parser for GLM4-MoE loader.
class Glm4MoeConfigParser : public BaseModelConfigParser {
 public:
  // Parse config from config.json
  virtual Status ParseModelConfig(const nlohmann::json &config_json,
                                  const ParallelismBasicConfig &parallel_basic_config, const std::string &model_dir,
                                  std::shared_ptr<BaseModelConfig> &model_config) override;

 private:
  // Parse MoE config
  Status ParseMoeConfig(const nlohmann::json &config_json, std::shared_ptr<Glm4MoeConfig> glm4_moe_config);

  // Parse quant config
  Status ParseQuantConfig(const nlohmann::json &config_json, std::shared_ptr<Glm4MoeConfig> glm4_moe_config,
                          const std::string &yaml_weight_quant_method, const std::string &yaml_gptq_backend);
};

}  // namespace ksana_llm
