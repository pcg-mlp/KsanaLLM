/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/base_model_config_parser.h"
#include "models/deepseek_v4/deepseek_v4_config.h"

namespace ksana_llm {

// Model config parser for deepseek_v4.
class DeepSeekV4ConfigParser : public BaseModelConfigParser {
 public:
  // Parse config from config.json
  virtual Status ParseModelConfig(const nlohmann::json &config_json,
                                  const ParallelismBasicConfig &parallel_basic_config, const std::string &model_dir,
                                  std::shared_ptr<BaseModelConfig> &model_config) override;

 private:
  // Parse quant config
  Status ParseQuantConfig(const nlohmann::json &quant_config_json, std::shared_ptr<DeepSeekV4Config> deepseek_v4_config,
                          const std::string &yaml_weight_quant_method, const std::string &yaml_gptq_backend);
};
}  // namespace ksana_llm
