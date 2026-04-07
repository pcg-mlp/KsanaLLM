/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/base/base_model_config_parser.h"

namespace ksana_llm {

struct ArcHunyuanVideoConfig;

// Model config parser for ArcHunyuanVideo loader.
class ArcHunyuanVideoConfigParser : public BaseModelConfigParser {
 public:
  // Parse config from config.json
  virtual Status ParseModelConfig(const nlohmann::json& config_json,
                                  const ParallelismBasicConfig& parallel_basic_config, const std::string& model_dir,
                                  std::shared_ptr<BaseModelConfig>& model_config) override;

 private:
  // Parse quant config from config.json and YAML.
  Status ParseQuantConfig(const nlohmann::json& config_json, std::shared_ptr<ArcHunyuanVideoConfig> arc_config,
                          const std::string& yaml_weight_quant_method);
};

}  // namespace ksana_llm
